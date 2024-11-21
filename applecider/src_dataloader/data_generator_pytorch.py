import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# for the tools stolen functions 
import pandas as pd
import random
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src_dataloader.data_preprocessor import SpectraProcessor


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- OrganizeData -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

class_dictionary = {'AGN': 1,
                    'Stellar variable': 2,
                    'SN Ia': 3,
                    'QSO': 4,
                    'SN Ic-SLSN': 5,
                    'Cataclysmic': 6,
                    'SN IIP': 7,
                    'Blazar': 8,
                    'SN Ia-91T': 9,
                    'SN IIn': 10,
                    'SN II': 11,
                    'SN Ib': 12,
                    'Novae': 13,
                    'SN II-pec': 14,
                    'SLSN-II': 15,
                    'SN Ia-91bg': 16,
                    'UV Ceti': 17,
                    'AM CVn': 18,
                    'SN Ibn': 19,
                    'U Gem': 20,
                    'LPV': 21,
                    'SN Ia-03fg': 22,
                    'SN Ic-BL': 23,
                    'afterglow': 24,
                    'Polars': 25,
                    'S Doradus': 26,
                    'Eclipsing': 27,
                    'YSO': 28,
                    'Galactic Nuclei': 29,
                    'Semiregular': 30,
                    'Mira': 31,
                    'RR Lyrae': 32,
                    'SLSN-I': 33,
                    'Cepheid': 34,
                    'Algol': 35,
                    'Pulsar': 36,
                    'FU Ori': 37,
                    'X-Ray Burster': 38,
                    'Pulsating': 39,
                    'SN Ca-rich': 40,
                    'Recurrent Nova': 41,
                    'SN Ib/c': 42,
                    'SN IIL': 43,
                    'BL Lac': 44,
                    'SN': 45,
                    'Seyfert': 46,
                    'SN Ia-18byg': 47,
                    'FBOT': 48,
                    'SN Ic': 49,
                    'SN I': 50,
                    'GRB': 51,
                    'SN Ib-p': 52,
                    'Nova-like': 53,
                    'SN Icn': 54,
                    'gravitational lensing': 55,
                    'SN Ia-02cx': 56,
                    'SN Ib-norm': 57,
                    'SN Ia-norm': 58,
                    'SN Ic-norm': 59,
                    'long GRB': 60,
                    'SN Ib-pec': 61,
                    'SN Ic.5-SLSN': 62,
                    'SN Ia-pec': 63,
                    'Solar System Object': 64,
                    'SN Ia-CSM': 65,
                    'Classical Nova': 66,
                    'Tidal Disruption Event': 67,
                    'SN IIb': 68,
                    'microlensing': 69,
                    'SN II-norm': 70,
                    'Anomolous': 71
                    
                }


class OrganizeData:
    
    def no_test_contamination_df(df, test_df):
        '''  ☆ removes object IDs in test dataframe so no overlap in the training set ☆ '''
        # test IDs to list
        test_object_ids = test_df['obj_id'].to_list()
        # what we will sample from 
        train_df = df[~df['obj_id'].isin(test_object_ids)]

        return train_df

    def sample_objects_from_df(df, type_col, new_col, n_test=20, n_train=40):
        '''
        Parameters
        ----------
        df : DataFrame
            ZTF ID, classification/type of object
        label_col : 'type'
        
        Returns
        ----------
        df but with new column that has numerical classifications from class_dictionary
        '''
        
        # test dataset
        test_df_total = df.sample(n=n_test, random_state=42)
        # randomly sample 1 object from every class
        test_df_random = df.sample(frac=1).drop_duplicates(type_col).sort_index()

        test_data_df = pd.concat([test_df_total, test_df_random])
        # replace classes with numerical values
        test_data_df[new_col] = test_data_df[type_col].map(class_dictionary)


        # train dataset
        train_data_df = OrganizeData.no_test_contamination_df(df, test_data_df)
        # sample n from df
        train_df_total = train_data_df.sample(n=n_train, random_state=42)
        # randomly sample 1 object from every class
        train_df_random = train_data_df.sample(frac=1).drop_duplicates(type_col).sort_index()
        train_data_df = pd.concat([train_df_total, train_df_random])
        # replace classes with numerical values
        train_data_df[new_col] = train_data_df[type_col].map(class_dictionary)
        train_data_df.reset_index(drop=True)

        return test_data_df.reset_index(drop=True),  train_data_df.reset_index(drop=True)
    
    
    def create_df_of_object_alerts_in_dataset(test_df, train_df, test_data_dir_path, train_data_dir_path, label_col):
        '''  ☆ creates df for testing and training sets that has the object IDs, alerts for each object ID, classification,
            and then the numerical label from class_dictionary  ☆
        
        Parameters
        ----------
        test_df : Dataframe
            object IDs in testing set
        train_df : dataframe
            object IDs in training set
        test_data_dir_path : 
            where the test set object alerts have been saved to 
        train_data_dir_path : 
            where the training set object alerts have been saved to 
        label col:
            name of the column with the numerical classifications

        Returns
        ----------
        test_data : Dataframe
            bject IDs, names of their alerts, real classification, and numerical classification label
        train_data : Dataframe
            object IDs, names of their alerts, real classification, and numerical classification label
        '''

        # test part
        test_data_files = [f for f in os.listdir(test_data_dir_path) if f.endswith('.npy')]
        test_data_names = [f.split('_')[0] for f in test_data_files]

        test_data = pd.DataFrame(test_data_names, columns=['name'])
        test_data['file'] = test_data_files
        test_data = test_data.merge(test_df[['obj_id','type', label_col]],
                                    left_on='name', right_on='obj_id', how='left')

        test_data = test_data.drop(columns=['obj_id'])
        test_data = test_data.sort_values(by='file')
        test_data = test_data.reset_index(drop=True)
        
        
        # train part
        train_data_files = [f for f in os.listdir(train_data_dir_path) if f.endswith('.npy')]
        train_data_names = [f.split('_')[0] for f in train_data_files]

        train_data = pd.DataFrame(train_data_names, columns=['name'])
        train_data['file'] = train_data_files
        train_data = train_data.merge(train_df[['obj_id','type', label_col]],
                      left_on='name', right_on='obj_id', how='left')

        train_data = train_data.drop(columns=['obj_id'])
        train_data = train_data.sort_values(by='file')
        train_data = train_data.reset_index(drop=True)
        
        return test_data, train_data
    
    
    def split_and_compute_class_weights(df, label_col, split_ratio=0.8, random_seed=42, nb=None, verbose=False):
        '''
        Train, Validation files and class weights for dataset
        Parameters
        ----------
        df : DataFrame
            dataframe with ZTF IDs, and classifications of each object
        label_col : 'type'
            name of your "type/classification" column in df 
        gp_wavelengths : numpy.ndarray
            Wavelengths to evaluate the Gaussian Process at.
        Returns
        ----------
        train files, validation files, class weights
        '''

        str_label = df[label_col].value_counts(dropna=False).keys().tolist()
        types_dict = { label_col : str_label}

        if nb is not None:
            df = df.groupby(label_col).head(nb)

        train_df_list, val_df_list = [], []
        unique_labels = df[label_col].unique()

        for label in unique_labels:
            df_filtered = df[df[label_col] == label]
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

        train_files = train_df['file'].tolist()
        val_files = val_df['file'].tolist()

        return train_files, val_files
    


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- DataGenerator -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, preprocessed_path, df, step, file_list=None, **kwargs):
        super().__init__(**kwargs)
        self.preprocessed_path = preprocessed_path
        self.step = step
        self.df = df

        if file_list is not None:
            self.data_files = file_list
        else:
            self.data_files = [f for f in os.listdir(preprocessed_path) if f.endswith('.npy')]
        
    def __len__(self): 
        return(len(self.data_files))
    
    def __getitem__(self, index):    
        ''' load processed object alerts to get photometry, metadata, images''' 
        file_path = os.path.join(self.preprocessed_path, str(self.data_files[index]))
        sample = np.load(file_path, allow_pickle=True).item()

        photometry = sample['photometry']
        metadata = sample['metadata'].to_numpy()
        images = sample['images']

        # get spectra csv, save wavelengths fluxes
        obj_id_alert = str(self.data_files[index])
        obj_id = obj_id_alert[:12] # only includes ZTFID from 'ZTFID_alerts.npy' 
        spectra_df = SpectraProcessor.read_spectra_csv(obj_id, '/data/dev/ml_skyportal/AJs_Stuff/(aj)data_all')
        spectra_df = spectra_df.astype(float) # to reassure us <3  

        # get label
        obj_df = self.df[self.df['name'] == obj_id]
        obj_label = obj_df[self.step].iloc[0]
        
        # convert photometry, metadata, images, spectra to tensors
        photometry_tensor = torch.tensor(photometry)
        metadata_tensor = torch.tensor(metadata)
        images_tensor = torch.tensor(images)
        spectra_tensor = torch.from_numpy(spectra_df.values)
        # convert label to tensor
        target = torch.tensor(obj_label, dtype=torch.int8)

        return photometry_tensor, metadata_tensor, images_tensor, spectra_tensor, target

