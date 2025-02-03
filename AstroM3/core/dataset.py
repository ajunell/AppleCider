import os

import joblib
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from scipy import stats


class DataGenerator(Dataset):

    def __init__(self, config, split='train'):
        super(DataGenerator, self).__init__()

        self.split = split
        self.preprocessed_path = config['preprocessed_path']
        self.spectra_path = config['spectra_path']
        self.step = config['step']
        self.random_seed = config['random_seed']
        self.classes = config['classes']
        self.max_samples = config['max_samples']
        self.scaler = joblib.load(config['scaler_path'])

        if self.split == 'train' or self.split == 'val':
            self.df = pd.read_csv(config['df_path'])
        else:
            # TODO Fix later
            raise ValueError('Split must be either train or val. FIX THIS LATER')

        self._filter_classes()
        self._limit_samples()
        self._remove_weird_spectra()
        self._split()

        # create convenient mapping for label from str to int and from int to str
        if config['group_labels']:
            self.id2target = {0: 'SN I', 1: 'SN II', 2: 'Cataclysmic', 3: 'AGN', 4: 'Tidal Disruption Event'}
            self.target2id = {'SN Ia': 0, 'SN II': 1, 'SN IIP': 1, 'Cataclysmic': 2, 'AGN': 3, 'SN IIn': 1, 'SN Ic': 0,
                              'SN Ib': 0, 'SN IIb': 1, 'Tidal Disruption Event': 4}
        else:
            self.id2target = {i: x for i, x in enumerate(sorted(self.df[self.step].unique()))}
            self.target2id = {v: k for k, v in self.id2target.items()}

        self.num_classes = len(self.id2target)

    def _filter_classes(self):
        """ filter classes if necessary """
        if self.classes:
            print(f'Left only with classes: {self.classes}')
            self.df = self.df[self.df[self.step].isin(self.classes)]

    def _limit_samples(self):
        """ downsample samples for each class if max_samples is set """
        if self.max_samples:
            for cls in self.df[self.step].unique():
                df_cls = self.df[self.df[self.step] == cls]
                df_not_cls = self.df[self.df[self.step] != cls]

                if len(df_cls) > self.max_samples:
                    print(f'Down sampled class {cls} from {len(df_cls)} to {self.max_samples}')
                    df_cls_down = df_cls.sample(n=self.max_samples, random_state=self.random_seed)
                    self.df = pd.concat([df_not_cls, df_cls_down], ignore_index=True)

    def _remove_weird_spectra(self):
        """ remove 'weird' spectra that significantly differs from other spectra """
        # spectra that have min wavelength higher than 4500
        min_names = ['ZTF19acykqyr', 'ZTF20aamttiw', 'ZTF19aavwbpc', 'ZTF18acclexy', 'ZTF23abnprwj', 'ZTF19acxysob',
                     'ZTF19aaabzpt', 'ZTF19aceckht', 'ZTF19aamfupk', 'ZTF19acxxwvi', 'ZTF22aaecchp', 'ZTF18acdvvgx',
                     'ZTF18acbzojv', 'ZTF22abahzeh', 'ZTF20abxzrqw', 'ZTF18acepwhb', 'ZTF20absvtnc', 'ZTF22aaeasul',
                     'ZTF19acgjnfz', 'ZTF20abxkwbi', 'ZTF18acxbksd', 'ZTF19acxowrr', 'ZTF19abdznxo', 'ZTF19abfwfei',
                     'ZTF18aabcdai', 'ZTF19aaxpjpq', 'ZTF19aavqoyu', 'ZTF19abiietd']

        # spectra that have max wavelength lower than 7980
        max_names = ['ZTF20abwzqzo', 'ZTF19abueupg', 'ZTF23aajrmfh', 'ZTF18acefuhk', 'ZTF18acsjdxo', 'ZTF19aautrth',
                     'ZTF18aceisbk', 'ZTF19aarykkb', 'ZTF20abawntz', 'ZTF19acvrjuw', 'ZTF18acpdvos', 'ZTF20acgrwej',
                     'ZTF18abwlupf', 'ZTF22aalvmic', 'ZTF20adadrhw', 'ZTF18aaxzcvd', 'ZTF21aceqrju', 'ZTF20abzcfqn',
                     'ZTF21abbyhvw', 'ZTF18acefxko', 'ZTF18abucxcj', 'ZTF18acrwheu', 'ZTF18acidntq', 'ZTF19aadnxat',
                     'ZTF18aaygmuq', 'ZTF23aavyhbo', 'ZTF23abtjkql', 'ZTF18absvcae', 'ZTF24aaczfve', 'ZTF18acbwasc',
                     'ZTF18aavpyfs', 'ZTF18aabcdai', 'ZTF19acryurj', 'ZTF21abljmmv', 'ZTF18acdwcvh', 'ZTF20aaertpj']

        self.df = self.df[~self.df['name'].isin(min_names)]
        self.df = self.df[~self.df['name'].isin(max_names)]

    def _split(self):
        """ split into train and val based on name """
        unique_names = self.df['name'].unique()
        train_names, val_names = train_test_split(unique_names, test_size=0.2, random_state=self.random_seed)

        if self.split == 'train':
            self.df = self.df[self.df['name'].isin(train_names)]
        else:
            self.df = self.df[self.df['name'].isin(val_names)]

    def __len__(self):
        return len(self.df)

    @staticmethod
    def read_spectra_csv(object_id, base_path):
        """ get wavelength, flux from spectra csv """
        file_path = os.path.join(base_path, object_id, 'spectra.csv')
        spectra_df = pd.read_csv(file_path)
        spectra_df = spectra_df[['wavelengths', 'fluxes']]

        spectra = spectra_df.to_numpy()
        spectra = spectra.astype(float)

        return spectra

    @staticmethod
    def preprocess_spectra(spectra):
        """ limit wavelength to 4500 - 7980, interpolate and normalize """
        new_wavelength = np.linspace(4500, 7980, 7980 - 4500 + 1)

        # remove nans from flux
        spectra = spectra[~np.isnan(spectra).any(axis=1)]

        f = interp1d(spectra[:, 0], spectra[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
        flux = f(new_wavelength)

        mean = np.mean(flux)
        mad = stats.median_abs_deviation(flux)
        flux = (flux - mean) / mad

        flux = flux.reshape((1, -1))
        flux = flux.astype(np.float32)

        return flux

    def __getitem__(self, index):
        """ load processed object alerts to get photometry, metadata, images """
        el = self.df.iloc[index]
        target = self.target2id[el[self.step]]

        file_path = os.path.join(self.preprocessed_path, el['file'])
        sample = np.load(file_path, allow_pickle=True).item()

        metadata = sample['metadata'].to_numpy()
        metadata = self.scaler.transform(metadata.reshape(1, -1))[0]
        metadata = metadata.astype(np.float32)

        images = sample['images']
        images = np.transpose(images, (2, 0, 1))
        images = images.astype(np.float32)

        # TODO add the last channel back
        photometry = sample['photometry'][:, 1:-1]

        # TODO create mask dynamically
        photometry_mask = torch.ones(len(photometry))

        spectra = self.read_spectra_csv(el['name'], self.spectra_path)
        spectra = self.preprocess_spectra(spectra)

        return photometry, photometry_mask, metadata, images, spectra, target
