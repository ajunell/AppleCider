import torch
from torch import nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from scipy import stats
import os
import joblib
import pickle
import random
from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight
    
    
def split_and_compute_class_weights(df, step, group_labels=False, save_files=True, save_train_files_path= None, save_val_files_path=None, save_class_weights_path = None, split_ratio=0.8, random_seed=42, nb=None, verbose=False):
    
    if group_labels:
        group_labels = {'SN Ia': 0, 'SN Ic': 0, 'SN Ib': 0, 'SN II': 1, 'SN IIP': 1, 'SN IIn': 1,
                        'SN IIb': 1, 'Cataclysmic': 2, 'AGN': 3, 'Tidal Disruption Event': 4}
        df.replace({step:group_labels})
   
    else:
        id2target = {'SN Ia':0 ,'SN Ic':1,  'SN Ib':2 , 'SN II': 3, 'SN IIP': 4, 'SN IIn': 5,
                    'SN IIb': 6, 'Cataclysmic': 7, 'AGN': 8, 'Tidal Disruption Event': 9}
        target2id = {v: k for k, v in id2target.items()}
        
        #df = df.replace({step: target2id})
        df = df.replace({step: id2target})
    
    if nb is not None:
        df = df.groupby(step).head(nb)

    train_df_list, val_df_list = [], []
    unique_labels = df[step].unique()

    for label in unique_labels:
        df_filtered = df[df[step] == label]
        unique_obj_ids = df_filtered['name'].unique()
        random.seed(random_seed)
        random.shuffle(unique_obj_ids)
        split_idx = int(len(unique_obj_ids) * split_ratio)
        train_obj_ids = unique_obj_ids[:split_idx]
        val_obj_ids = unique_obj_ids[split_idx:]
        train_df_list.append(df_filtered[df_filtered['name'].isin(train_obj_ids)])
        val_df_list.append(df_filtered[df_filtered['name'].isin(val_obj_ids)])
    
    train_df = pd.concat(train_df_list).reset_index(drop=True)
    val_df = pd.concat(val_df_list).reset_index(drop=True)

    train_obj_ids = train_df['name'].unique()
    val_obj_ids = val_df['name'].unique()

    assert len(set(train_obj_ids).intersection(set(val_obj_ids))) == 0

    class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=train_df[step])
   
    class_weight_dict = dict(zip(unique_labels, class_weights))

    train_files = train_df['file'].tolist()
    val_files = val_df['file'].tolist()

    if verbose:
        print_types(train_df, columns=[label_col])
        print_types(val_df, columns=[label_col])
        
        
    if save_files:
        with open(os.path.join(save_train_files_path), 'wb') as file:
            pickle.dump(train_files, file)
            print(f"saved train files to {save_train_files_path}.")
            
        with open(os.path.join(save_val_files_path), 'wb') as file:
            pickle.dump(val_files, file)
            print(f"saved val files to {save_val_files_path}.")
            
        
        with open(os.path.join(save_class_weights_path), 'wb') as file:
            pickle.dump(weights, file)
            print(f"saved weights to {save_class_weights_path}.")
    
    return train_files, val_files, class_weight_dict



class DataGenerator(Dataset):

    def __init__(self, config, split='train'):
        super(DataGenerator, self).__init__()

        self.split = split
        self.preprocessed_path = config['preprocessed_path']
        self.step = config['step']
        self.random_seed = config['random_seed']
        self.classes = config['classes']
        self.max_samples = config['max_samples']
        self.mode = config['mode']
        self.group_labels = config['group_labels']
        
        self.generate_train_val_files = config['generate_train_val_files']
        self.train_files = config['train_files_path']
        self.val_files = config['val_files_path']
        self.weights = config['class_weights_path']

        if self.mode == 'meta' or self.mode == 'all':
            self.scaler = joblib.load(config['scaler_path'])
            if self.scaler is None:
                raise ValueError('No scaler path. Add path.')
         
        if self.split == 'train' or self.split == 'val':
            self.df = pd.read_csv(config['df_path'])
    
        else:
            ## TODO Fix later
            raise ValueError('Split must be either train or val.')
        
        self._split()

        ## create convenient mapping for label from str to int and from int to str
        if self.group_labels:
            self.id2target = {0: 'SN I', 1: 'SN II', 2: 'Cataclysmic', 3: 'AGN', 4: 'Tidal Disruption Event'}
            self.target2id = {'SN Ia': 0 , 'SN Ic': 0,  'SN Ib': 0, 'SN II': 1, 'SN IIP': 1, 'SN IIn': 1, 'SN IIb': 1, 'Cataclysmic': 2, 'AGN': 3, 'Tidal Disruption Event': 4}
        else:
            
            self.id2target = {'SN Ia':0 ,'SN Ic':1,  'SN Ib':2 , 'SN II': 3, 'SN IIP': 4, 'SN IIn': 5,
                 'SN IIb': 6, 'Cataclysmic': 7, 'AGN': 8, 'Tidal Disruption Event': 9}
            self.target2id = {v: k for k, v in self.id2target.items()}

        self.num_classes = len(self.id2target)

    def _split(self):
        """ sort train, val based on alert names in already created pkl from preprocessing steps """
        
        ## for pre-saved train, validation files
        if os.path.isfile(self.train_files) and os.path.isfile(self.val_files):
            
            if self.split == 'train':
                with open(self.train_files, 'rb') as file:
                    train_files_saved = pickle.load(file)
                    self.df = self.df[self.df['file'].isin(train_files_saved)]   
            elif self.split == 'val':
                with open(self.val_files, 'rb') as file:
                    val_files_saved = pickle.load(file) #group_labels
                    self.df = self.df[self.df['file'].isin(val_files_saved)]
            else:
                print("Something went wrong with train, val split.")
        
        else: ## w/o pre-saved train, val files, generate & optionally save them + class weights
            try:
                train_files, val_files, class_weight_dict = split_and_compute_class_weights(self.df, self.step, group_labels=self.group_labels, save_files=self.generate_train_val_files, save_train_files_path=self.train_files, save_val_files_path=self.val_files, class_weights_path=self.weights)
                if self.split == 'train':
                    self.df = self.df[self.df['file'].isin(train_files)]
                
                elif self.split == 'val':
                    self.df = self.df[self.df['file'].isin(val_files)]
                else:
                    raise ValueError("uhhh something happened with split_compute_class_weights, try again?")
            
            except NameError:
                raise NameError(f"NameError: {self.train_files} and {self.val_files} DO NOT exist!\n You need to generate train, val files. Fix train, val paths or create files with config['generate_train_val_files'] = True.")
                                                                      
                                                                      
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """ load processed object alerts to get photometry, metadata, images, spectra """
        
        el = self.df.iloc[index]
        target = self.target2id[el[self.step]]

        file_path = os.path.join(self.preprocessed_path, el['file'])
        sample = np.load(file_path, allow_pickle=True).item()
        
         ## photometry formating: mjd, ztf-g, ztf-r, ztf-i
        photometry = sample['photometry']
        photometry_tensor = torch.tensor(photometry, dtype=torch.float32)
        photo_len = len(photometry_tensor)
        max_photo = 230  ## maximum photometry length from an alert
        ## padded photometry
        if photo_len < max_photo:
            photometry_padded = nn.ConstantPad1d((0, 0, 0, max_photo - photo_len), 0)(photometry_tensor)
        else: 
            raise ValueError("Reset max photometry length") 

        metadata = sample['metadata'].to_numpy()
        if self.mode == 'meta' or self.mode == 'all':
            metadata = self.scaler.transform(metadata.reshape(1, -1))[0]
            metadata = metadata.astype(np.float32)
        metadata = torch.tensor(metadata)

        images = sample['images']
        images = np.transpose(images, (2, 0, 1))
        images = images.astype(np.float32)
        images = torch.tensor(images)
        
        spectra = sample['spectra']
        spectra = torch.tensor(spectra)

        return photometry_padded, metadata, images, spectra, target