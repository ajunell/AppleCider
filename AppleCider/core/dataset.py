import os

import joblib
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from scipy import stats

import pickle
from torch import nn


from tqdm.auto import tqdm


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
        
        self.train_files = config['train_files_path']
        self.val_files = config['val_files_path']

        if self.split == 'train' or self.split == 'val':
            self.df = pd.read_csv(config['df_path'])
        else:
            # TODO Fix later
            raise ValueError('Split must be either train or val. FIX THIS LATER')
        
        self._split()

        ## create convenient mapping for label from str to int and from int to str
        if config['group_labels']:
            self.id2target = {0: 'SN I', 1: 'SN II', 2: 'Cataclysmic', 3: 'AGN', 4: 'Tidal Disruption Event'}
            self.target2id = {'SN Ia': 0 , 'SN Ic': 0,  'SN Ib': 0, 'SN II': 1, 'SN IIP': 1, 'SN IIn': 1, 'SN IIb': 1, 'Cataclysmic': 2, 'AGN': 3, 'Tidal Disruption Event': 4}
        else:
            self.id2target = {i: x for i, x in enumerate(sorted(self.df[self.step].unique()))}
            self.target2id = {v: k for k, v in self.id2target.items()}

        self.num_classes = len(self.id2target)

    def _split(self):
        """ sort train, val based on alert names in already created pkl from preprocessing steps """

        if self.split == 'train':
            with open(self.train_files, 'rb') as file:
               train_files = pickle.load(file)            
            self.df = self.df[self.df['file'].isin(train_files)]
        
        elif self.split == 'val':
            with open(self.val_files, 'rb') as file:
               val_files = pickle.load(file) 
            self.df = self.df[self.df['file'].isin(val_files)]
        else:
            print("uhhhh hello??? where are your files")

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

        ## photometry formating: mjd, ztf-g, ztf-r, ztf-i
        photometry = sample['photometry']
        photometry_tensor = torch.tensor(photometry, dtype=torch.float32)
        photo_len = len(photometry_tensor)
        max_photo = 225                      ## maximum photometry length from an alert
        add_dim = max_photo - photo_len

        ## padded photometry so all photometry the same length
        if photo_len <= 225:
            photometry_padded = nn.ConstantPad1d((0, 0, 0, add_dim), 0)(photometry_tensor)
        else:
            # check max photo length from alerts again! 
            print("too much photometry. try again!", photo_len) 
        
        photometry_mask = torch.ones((photometry_padded.size(0), photometry_padded.size(1)))
        
        spectra = sample['spectra']
        
        metadata = torch.tensor(metadata)
        spectra = torch.tensor(spectra)
        images = torch.tensor(images)

        return photometry_padded, photometry_mask, metadata, images, spectra, target