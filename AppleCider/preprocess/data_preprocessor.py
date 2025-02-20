import os
import warnings
import numpy as np
import pandas as pd
import multiprocessing
import gzip
import io
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from scipy.interpolate import interp1d
from scipy import stats

from tqdm import tqdm
import pickle
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# add in pad_sequence package
from torch.nn.utils.rnn import pad_sequence


class AlertProcessor:
    """ ☆ procces object's alert package ☆ (see arXiv:1902.02227 for more info) """
    
    @staticmethod
    def get_alerts(base_path, obj_id):
        return np.load(os.path.join(base_path, obj_id, 'alerts.npy'), allow_pickle=True)

    @staticmethod
    def process_image(data, normalize=True):
        ''' returns processed image as a 63x63 np array '''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=AstropyWarning)
            warnings.simplefilter('ignore')
            with gzip.open(io.BytesIO(data), "rb") as f:
                image = np.nan_to_num(fits.open(io.BytesIO(f.read()), ignore_missing_end=True)[0].data)
        if normalize:
            norm = np.linalg.norm(image)
            if norm != 0:
                image /= norm
        return np.pad(image, [(0, 63 - s) for s in image.shape], mode="constant", constant_values=1e-9)[:63, :63]

    @staticmethod
    def process_alert(alert):
        ''' process metadata, images from alerts '''
        metadata = alert['candidate']
        metadata_df = pd.DataFrame([metadata])
        metadata_df['obj_id'] = alert['objectId']

        cutout_dict = {
            cutout: AlertProcessor.process_image(alert[f"cutout{cutout.capitalize()}"]["stampData"])
            for cutout in ("science", "template", "difference")
        }
        assembled_image = np.zeros((63, 63, 3))
        assembled_image[:, :, 0] = cutout_dict["science"]
        assembled_image[:, :, 1] = cutout_dict["template"]
        assembled_image[:, :, 2] = cutout_dict["difference"]

        return metadata_df, assembled_image


    @staticmethod
    def get_process_alerts(obj_id, base_path):
        
        alerts = AlertProcessor.get_alerts(base_path, obj_id)
        metadata_list = []
        images = []

        for alert in alerts:
            metadata_df, image = AlertProcessor.process_alert(alert)
            metadata_list.append(metadata_df)
            images.append(image)

        return pd.concat(metadata_list, ignore_index=True), images
    

    @staticmethod
    def select_alerts(data, max_alerts=6):
        
        ''' sample from maximum of XYZ alerts '''
        def sample_alerts(alerts):
            num_alerts = len(alerts)
            if num_alerts <= max_alerts:
                return alerts
            selected_alerts = [alerts[0], alerts[-1]]
            if num_alerts > 2:
                step = (num_alerts - 2) / (max_alerts - 2)
                selected_alerts += [alerts[int(step * i + 1)] for i in range(max_alerts - 2)]
            return selected_alerts

        data_by_obj_id = {}
        for sample in data:
            obj_id = sample['obj_id']
            if obj_id not in data_by_obj_id:
                data_by_obj_id[obj_id] = []
            data_by_obj_id[obj_id].append(sample)

        selected_data = []
        for obj_id, alerts in data_by_obj_id.items():
            alerts_sorted = sorted(alerts, key=lambda x: x['alerte'])
            selected_data.extend(sample_alerts(alerts_sorted))

        return selected_data



class PhotometryProcessor:
    
    """ ☆ procces object's photometry, metadata """
    
    @staticmethod
    def clean_photometry(df, df_type):
        ''' cleans photometry dataframe '''
        df = PhotometryProcessor.clean_dataframe(df)
        df['type'] = df_type[df_type['obj_id'] == df['obj_id'].iloc[0]]['type'].values[0]
        df.dropna(subset=['mag', 'magerr'], inplace=True)
        return df.reset_index(drop=True)
    
    @staticmethod
    def clean_dataframe(df):
        ''' renames columns, converts jd to MJD  '''
        df = df.rename(columns={
            'magpsf': 'mag',
            'sigmapsf': 'magerr',
            'fid': 'filter',
            'scorr': 'snr',
            'diffmaglim': 'limiting_mag' })
        df['filter'] = df['filter'].replace({1: 'ztfg', 2: 'ztfr', 3: 'ztfi'})
        ## remove i filter
        df = df[df['filter'] != 'ztfi']
        df['mjd'] = df['jd'] - 2400000.5
        df = df[['obj_id', 'jd', 'mjd', 'mag', 'magerr', 'snr', 'limiting_mag', 'filter']]
        return df

    @staticmethod
    def process_csv(object_id, df_bts, base_path):   
        ''' creates file path for photometry.csv, cleans photometry'''
        file_path = os.path.join(base_path, object_id, 'photometry.csv')
        return PhotometryProcessor.clean_photometry(pd.read_csv(file_path), df_bts) if os.path.exists(file_path) else pd.DataFrame()

    @staticmethod
    def get_first_valid_index(df, min_points=1):
        '''counts occurences of each filter, finds index that meets minimum number of points in each filter'''
        filter_counts = {'ztfr': 0, 'ztfg': 0, 'ztfi':0}
        for i in range(len(df)):
            current_filter = df['filter'].iloc[i]
            if current_filter in filter_counts:
                filter_counts[current_filter] += 1
                if filter_counts[current_filter] >= min_points:
                    return i
        return -1

    @staticmethod
    def add_metadata_to_photometry(photo_df, metadata_df):
        ''' cleans metadata, merges photometry_df with metadata_df'''
        
        metadata_df_copy = PhotometryProcessor.clean_dataframe(metadata_df.copy())
        df = pd.merge(photo_df, metadata_df_copy, on=['obj_id', 'jd', 'mjd', 'mag', 'magerr', 'snr', 'limiting_mag', 'filter'], how='outer', suffixes=('', '_metadata'))        
        df = df[['obj_id', 'jd', 'mjd', 'mag', 'magerr', 'snr', 'limiting_mag', 'filter', 'type']]
        df['obj_id'] = df['obj_id'].ffill().bfill()
        df['type'] = df['type'].ffill().bfill()
        df = df.drop_duplicates(subset=['mjd', 'filter'], keep='first')
        df = df.sort_values(by=['mjd'])
        df.reset_index(drop=True, inplace=True)
        return df
    
    def find_valid_alert_index(df):
        for index, row in df.iterrows():
            if row['flux_ztfg'] == 0 and row['flux_ztfr'] == 0 and row['flux_ztfi'] == 0:
                return index - 1
        
        return len(df)
    
    def normalize_light_curve(df):
        #valid_index = find_valid_alert_index(df)
        #flux_data = df.loc[:valid_index, ['flux_ztfg', 'flux_ztfr', 'flux_ztfi']]
        #scaler = StandardScaler() # standardizes by removing mean, scaling to unit variance
        #normalized_flux = scaler.fit_transform(flux_data)
        #df.loc[:valid_index, ['flux_ztfg', 'flux_ztfr','flux_ztfi']] = normalized_flux
        
        flux_data = df.loc[:len(df), ['flux_ztfg', 'flux_ztfr', 'flux_ztfi']]
        scaler = StandardScaler() # standardizes by removing mean, scaling to unit variance
        normalized_flux = scaler.fit_transform(flux_data)
        df.loc[:len(df), ['flux_ztfg', 'flux_ztfr','flux_ztfi']] = normalized_flux
  
        return df
    

class SpectraProcessor:
    
    """ ☆ procces object's spectra (not in alerts.npy) ☆ """
    
    @staticmethod
    def get_spectra_df(object_id, base_path):
        ''' for when we want all of the columns in spectra.csv '''
        file_path = os.path.join(base_path, object_id, 'spectra.csv')
        spectra_df = pd.read_csv(file_path)
        return spectra_df
    
    @staticmethod
    def read_spectra_csv(object_id, base_path):
        """ get wavelength, flux from spectra csv """
        file_path = os.path.join(base_path, object_id, 'spectra.csv')
        spectra_df = pd.read_csv(file_path)
        spectra_df = spectra_df[['wavelength', 'flux']]
      
        return spectra_df
    
    @staticmethod
    def preprocess_spectra(spectra):
        """ limit wavelength to 4500 - 7980, interpolate and normalize """
        
        spectra = spectra.to_numpy()
        spectra = spectra.astype(float)
        
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

    
class DataSorter:
    """ ☆ filter out objects w/o SEDM spectra, split train test sets and save alert names ☆ """
    
    def get_obj_wSEDM_spectra(obj_id_list, data_dir):
        ''' ☆ list of object ids that have SEDM spectra ☆ '''
        
        obj_sedm_list = []
        
        for object_id in tqdm(obj_id_list, desc='Checking for SEDM spectra', leave=True):
            spectra_path = os.path.join(data_dir, object_id, 'spectra.csv')
            
            if os.path.isfile(spectra_path):
                spectra_df = pd.read_csv(os.path.join(data_dir, object_id, 'spectra.csv'))
                # check if spectra from Fritz (which has all these columns)
                if {'instrument_name', 'telescope_name', 'data_length'}.issubset(spectra_df.columns):
                    instrument = spectra_df['instrument_name'][0]
                    if instrument == 'SEDM':
                        obj_sedm_list.append(object_id)
    
        return obj_sedm_list

    def remove_test_data(df, test_df):
        ''' ☆ remove objects from test dataset ☆ '''
        test_object_ids = test_df['obj_id'].to_list()
        train_df = df[~df['obj_id'].isin(test_object_ids)]
        return train_df

    def sample_objects_from_df(df, type_col, class_list, data_dir, n_test=20, n_train=40, sedm_spec_only=True, downsample_SNIa=True):
        
        if sedm_spec_only:
            obj_with_sedm_spec = DataSorter.get_obj_wSEDM_spectra(df['obj_id'].to_list(), data_dir)
            ## remove objects without sedm spectra
            df = df[df['obj_id'].isin(obj_with_sedm_spec)]
        if downsample_SNIa: # redunant since dataset.py does this
            sn_ia = df[df['type'] == 'SN Ia'].sample(n=600, random_state=42)
            df = pd.concat([df[df['type'] != 'SN Ia'], sn_ia])
    
        ## remove objects not in the acceptable class list (class_list)
        df = df[df[type_col].isin(class_list)]
        
        ## TEST
        test_df_total = df.sample(n=n_test, random_state=42)
        ## randomly sample 1 object from every class
        test_df_random = df.sample(frac=1).drop_duplicates(type_col).sort_index()
        ## force sample more TDE:
        TDE_ = ['Tidal Disruption Event'] ; SN_ = ['SN IIb'] ; SN__ = ['SN Ib']
        test_df_TDE = df.query(f'type=={TDE_}').sample(n=8) ; test_df_SNIb = df.query(f'type=={SN__}').sample(n=8) ; test_df_SNIIb = df.query(f'type=={SN_}').sample(n=8)
        test_data_df = pd.concat([test_df_total, test_df_random, test_df_TDE, test_df_SNIb, test_df_SNIIb])
        test_data_df = test_data_df.drop_duplicates('obj_id')
    
        ## TRAIN
        train_data_df = DataSorter.remove_test_data(df, test_data_df)
        train_data_df = train_data_df.drop_duplicates('obj_id')
        train_data_df.reset_index(drop=True)
    
        return test_data_df.reset_index(drop=True),  train_data_df.reset_index(drop=True)
    
    
    def create_df_of_object_alerts_in_dataset(test_df, train_df, test_data_dir, train_data_dir):
        '''  ☆ creates df for testing and training sets that has the object IDs, alerts for each object ID, classification ☆
        
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
        test_data_files = [f for f in os.listdir(test_data_dir) if f.endswith('.npy')]
        test_data_names = [f.split('_')[0] for f in test_data_files]

        test_data = pd.DataFrame(test_data_names, columns=['name'])
        test_data['file'] = test_data_files
        test_data = test_data.merge(test_df[['obj_id','type']],
                                    left_on='name', right_on='obj_id', how='left')

        test_data = test_data.drop(columns=['obj_id'])
        test_data = test_data.sort_values(by='file')
        test_data = test_data.reset_index(drop=True)
        
        # train part
        train_data_files = [f for f in os.listdir(train_data_dir) if f.endswith('.npy')]
        train_data_names = [f.split('_')[0] for f in train_data_files]

        train_data = pd.DataFrame(train_data_names, columns=['name'])
        train_data['file'] = train_data_files
        train_data = train_data.merge(train_df[['obj_id','type']],
                      left_on='name', right_on='obj_id', how='left')

        train_data = train_data.drop(columns=['obj_id'])
        train_data = train_data.sort_values(by='file')
        train_data = train_data.reset_index(drop=True)
        
        return test_data, train_data
    
    
    
    def split_train_validation_files(df, label_col, split_ratio=0.8, random_seed=42, nb=None, verbose=False):
        '''
        Train, Validation files and class weights for dataset
        Parameters
        ----------
        df : DataFrame
            dataframe with ZTF IDs, and classifications of each object
        label_col : 'type'
            name of your "type/classification" column in df 
        Returns
        ----------
        train files, validation files
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


class DataPreprocessor:
    
    """ ☆ additional pre-processing of photometry, metadata ☆ """
    
    @staticmethod
    def Mag2Flux(df):
        ''' converts magnitude to flux'''
        df_copy = df.dropna().copy()
        df_copy['flux'] = 10 ** (-0.4 * (df_copy['mag'] - 23.9))
        df_copy['flux_error'] = (df_copy['magerr'] / (2.5 / np.log(10))) * df_copy['flux']
        df_copy = df_copy[['obj_id', 'mjd', 'flux', 'flux_error', 'filter', 'type', 'jd']]
        return df_copy
    
    @staticmethod    
    def Normalize_mjd(df):
        ''' normalize modified julian date'''
        df_copy = df.copy()
        df_copy['mjd'] = df_copy.groupby('obj_id')['mjd'].transform(lambda x: x - np.min(x))
        df_copy.reset_index(drop=True, inplace=True)
        return df_copy
    
    @staticmethod
    def convert_photometry(photo_df):
        ''' converts magnitude to flux, normalizes modifed Julian date of photometry df '''
        df_gp_ready = DataPreprocessor.Mag2Flux(photo_df)
        df_gp_ready = DataPreprocessor.Normalize_mjd(df_gp_ready).drop_duplicates().reset_index(drop=True)
        return df_gp_ready

    @staticmethod
    def cut_photometry(photo_df, metadata_df, index, max_mjd=10):    
        ''' ensure mjd max not exceeded'''
        jd_current = metadata_df['jd'].iloc[index]
        photometry_filtered = photo_df[photo_df['jd'] <= jd_current]
        return None if photometry_filtered['mjd'].max() > max_mjd else photometry_filtered

    @staticmethod
    def preprocess_metadata(metadata_df):
        ''' removes metadata duplicates and irrelevant columns '''
        metadata_df = metadata_df.drop_duplicates(subset=['jd'], keep='first')
        columns_metadata = [ "sgscore1", "sgscore2", "distpsnr1", "distpsnr2", "ra", "dec", "nmtchps", "sharpnr", "scorr", "sky", 'jd' ]
        return metadata_df[columns_metadata].fillna(-999.0)