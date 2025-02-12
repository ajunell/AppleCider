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
        self.step = config['step']
        self.random_seed = config['random_seed']
        self.classes = config['classes']
        self.max_samples = config['max_samples']
        self.scaler = joblib.load(config['scaler_path'])
        #self.scaler = config['scaler_path']
     
        # 2/10: save files 
        #self.train_files = config['save_train_files']
        #self.val_files = config['save_val_files']

        if self.split == 'train' or self.split == 'val':
            self.df = pd.read_csv(config['df_path'])
        else:
            # TODO Fix later
            raise ValueError('Split must be either train or val. FIX THIS LATER')

        self._filter_classes()
        self._limit_samples()
        self._split()

        # create convenient mapping for label from str to int and from int to str
        if config['group_labels']:
            self.id2target = {0: 'SN I', 1: 'SN II', 2: 'Cataclysmic', 3: 'AGN', 4: 'Tidal Disruption Event'}
            self.target2id = {'SN Ia': 0 , 'SN Ic': 0,  'SN Ib': 0, 'SN II': 1, 'SN IIP': 1, 'SN IIn': 1, 'SN IIb': 1, 'Cataclysmic': 2, 'AGN': 3, 'Tidal Disruption Event': 4}
        
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

    def _split(self):
        """ split into train and val based on name """
        unique_names = self.df['name'].unique()
        train_names, val_names = train_test_split(unique_names, test_size=0.2, random_state=self.random_seed)

        if self.split == 'train':
            self.df = self.df[self.df['name'].isin(train_names)]
            # save train file names
            #joblib.dump(train_names, self.train_files)
        else:
            self.df = self.df[self.df['name'].isin(val_names)]
            # save val file names
            #joblib.dump(val_names, self.val_files)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """ load processed object alerts to get photometry, metadata, images, spectra """
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
        ## removed mjd, flux error in pre-processing steps 
        photometry = sample['photometry']
        
        # TODO create mask dynamically
        photometry_mask = torch.ones(len(photometry))
        
        spectra = sample['spectra']

        return photometry, photometry_mask, metadata, images, spectra, target
