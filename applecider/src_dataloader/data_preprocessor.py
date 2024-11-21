import os
import warnings
import numpy as np
import pandas as pd
import multiprocessing
import gzip
import io
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from tqdm import tqdm
import pickle
import random

import src_dataloader.gaussian_process as gp

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --  Alert Processor -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
class AlertProcessor:
    ''' ☆ procces each object's alert package from ZTF ☆ (see arXiv:1902.02227 for more about alert distribution system) '''
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
    def select_alerts(data, max_alerts=30):
        ''' sample from maximum of 30 alerts '''
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


#  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --  Photometry Processor -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

class PhotometryProcessor:
    @staticmethod
    def clean_photometry(df, df_type, is_i_filter=True):
        ''' cleans photometry dataframe '''
        df = PhotometryProcessor.clean_dataframe(df, is_i_filter)
        df['type'] = df_type[df_type['obj_id'] == df['obj_id'].iloc[0]]['type'].values[0]
        df.dropna(subset=['mag', 'magerr'], inplace=True)
        return df.reset_index(drop=True)
    
    @staticmethod
    def clean_dataframe(df, is_i_filter=True):
        ''' cleans dataframe '''
        df = df.rename(columns={
            'magpsf': 'mag',
            'sigmapsf': 'magerr',
            'fid': 'filter',
            'scorr': 'snr',
            'diffmaglim': 'limiting_mag'
        })
        df['filter'] = df['filter'].replace({1: 'ztfg', 2: 'ztfr', 3: 'ztfi'})
        if not is_i_filter:
            df = df[df['filter'] != 'ztfi']
        df['mjd'] = df['jd'] - 2400000.5
        df = df[['obj_id', 'jd', 'mjd', 'mag', 'magerr', 'snr', 'limiting_mag', 'filter']]
        return df

    @staticmethod
    def process_csv(object_id, df_bts, base_path, is_i_filter=True):
        '''creates file path for photometry.csv'''
        file_path = os.path.join(base_path, object_id, 'photometry.csv')
        return PhotometryProcessor.clean_photometry(pd.read_csv(file_path), df_bts, is_i_filter) if os.path.exists(file_path) else pd.DataFrame()

    @staticmethod
    def get_first_valid_index(df, min_points=3):
        '''counts occurences of each filter, finds index that meets minimum number of points in each filter'''
        filter_counts = {'ztfr': 0, 'ztfg': 0, 'ztfi': 0}
        for i in range(len(df)):
            current_filter = df['filter'].iloc[i]
            if current_filter in filter_counts:
                filter_counts[current_filter] += 1
                if filter_counts[current_filter] >= min_points:
                    return i
        return -1

    @staticmethod
    def add_metadata_to_photometry(photo_df, metadata_df, is_i_filter=True):
        ''' cleans metadata, merges photometry_df with metadata_df'''
        metadata_df_copy = PhotometryProcessor.clean_dataframe(metadata_df.copy(), is_i_filter)
        df = pd.merge(photo_df, metadata_df_copy, on=['obj_id', 'jd', 'mjd', 'mag', 'magerr', 'snr', 'limiting_mag', 'filter'], how='outer', suffixes=('', '_metadata'))        
        df = df[['obj_id', 'jd', 'mjd', 'mag', 'magerr', 'snr', 'limiting_mag', 'filter', 'type']]
        df['obj_id'] = df['obj_id'].ffill().bfill()
        df['type'] = df['type'].ffill().bfill()
        df = df.drop_duplicates(subset=['mjd', 'filter'], keep='first')
        df = df.sort_values(by=['mjd'])
        df.reset_index(drop=True, inplace=True)
        return df


class SpectraProcessor:
    
    def get_spectra_df(object_id, base_path):
        ''' for when we want all of the columns in spectra.csv '''
        file_path = os.path.join(base_path, object_id, 'spectra.csv')
        spectra_df = pd.read_csv(file_path)
        return spectra_df
    
    def read_spectra_csv(object_id, base_path):
        ''' makes df from spectra.csv for list of objects '''
        file_path = os.path.join(base_path, object_id, 'spectra.csv')
        spectra_df = pd.read_csv(file_path)
        spectra_df = spectra_df.rename(columns={'ZTFID': 'obj_id'})
        spectra_df = spectra_df[['wavelengths', 'fluxes']]
        return spectra_df



# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- DataPreprocessor -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
class DataPreprocessor:
    #@staticmethod
    def Mag2Flux(df):
        ''' converts magnitude to flux'''
        df_copy = df.dropna().copy()
        df_copy['flux'] = 10 ** (-0.4 * (df_copy['mag'] - 23.9))
        df_copy['flux_error'] = (df_copy['magerr'] / (2.5 / np.log(10))) * df_copy['flux']
        df_copy = df_copy[['obj_id', 'mjd', 'flux', 'flux_error', 'filter', 'type', 'jd']]
        return df_copy
        
    def Normalize_mjd(df):
        ''' normalize modified julian date'''
        df_copy = df.copy()
        df_copy['mjd'] = df_copy.groupby('obj_id')['mjd'].transform(lambda x: x - np.min(x))
        df_copy.reset_index(drop=True, inplace=True)
        return df_copy
    
    def convert_photometry(photo_df):
        ''' converts magnitude to flux, normalizes modifed Julian date of photometry df '''
        df_gp_ready = DataPreprocessor.Mag2Flux(photo_df)
        df_gp_ready = DataPreprocessor.Normalize_mjd(df_gp_ready).drop_duplicates().reset_index(drop=True)
        return df_gp_ready

    @staticmethod
    def cut_photometry(photo_df, metadata_df, index, max_mjd=200):
        ''' ensure mjd max not exceeded'''
        jd_current = metadata_df['jd'].iloc[index]
        photometry_filtered = photo_df[photo_df['jd'] <= jd_current]
        return None if photometry_filtered['mjd'].max() > max_mjd else photometry_filtered

    @staticmethod
    def preprocess_metadata(metadata_df):
        ''' removes metadata duplicates and irrelevant columns '''
        metadata_df = metadata_df.drop_duplicates(subset=['jd'], keep='first')
        columns_metadata = [ "sgscore1", "sgscore2", "distpsnr1", "distpsnr2", "ra", "dec", "nmtchps", 
                                "sharpnr", "scorr", "sky", 'jd' ]
        return metadata_df[columns_metadata].fillna(-999.0)

    def process_and_save_sample(args):
        ''' save dictionary w/processed photometry, metadata, images to .npy at desired path '''
        res_dict = {}
        sample, save_dir, kernel, is_i_filter = args
        obj_id = sample['obj_id']
        alerte = sample['alerte']

        save_path = os.path.join(save_dir, f"{obj_id}_alert_{alerte}.npy")
        if os.path.exists(save_path):
            return
        gp_ready = sample['photometry']
        
        # remove filters w/w/less than 3 entries
        def remove_filter(photo_df):
            filters = photo_df['filter'].unique()
            for filt in filters:
                if len(photo_df[photo_df['filter'] == filt]) < 3:
                    photo_df = photo_df[photo_df['filter'] != filt]
            photo_df = photo_df.reset_index(drop=True)
            return photo_df
        
        gp_ready = remove_filter(gp_ready)
        if len(gp_ready) == 0:
            return

        gp_final = gp.process_gaussian(gp_ready, kernel=kernel, number_gp=180)

        columns = ['flux_ztfg', 'flux_error_ztfg', 'flux_ztfr', 'flux_error_ztfr']
        if is_i_filter:
            columns += ['flux_ztfi', 'flux_error_ztfi']
    
        for col in columns:
            if col not in gp_final.columns:
                gp_final[col] = 0.
                if not is_i_filter:
                    return
                
        def normalize_light_curve(df):
            df = df.copy()
            
            def find_valid_alert_index(df):
                for index, row in df.iterrows():
                    if row['flux_ztfg'] == 0 and row['flux_ztfi'] == 0 and row['flux_ztfr'] == 0:
                        return index - 1
                return len(df)
            
            valid_index = find_valid_alert_index(df)
            flux_data = df.loc[:valid_index, ['flux_ztfg', 'flux_ztfi', 'flux_ztfr']]
            scaler = StandardScaler() # standardizes by removing mean, scaling to unit variance
            normalized_flux = scaler.fit_transform(flux_data)
            df.loc[:valid_index, ['flux_ztfg', 'flux_ztfi', 'flux_ztfr']] = normalized_flux
            return df

        gp_final = normalize_light_curve(gp_final)

        useful_columns = ['mjd', 'flux_ztfg', 'flux_ztfr']
        if is_i_filter:
            useful_columns += ['flux_ztfi']
        sequences = gp_final[useful_columns].values
        padded_sequences = pad_sequences(sequences, padding='post', dtype='float32')

        res_dict.update({
            'obj_id': obj_id,
            'photometry': padded_sequences,
            'metadata': sample['metadata'],
            'images': sample['images'],
            'type_step1': sample['type_step1'],
            'target': sample['target'],
            'alerte': alerte})

        np.save(save_path, res_dict)





class TransientDataset():
    '''
    Parameters
    ----------
    preprocessed_path : processed samples directory
        where the sampled alerts from objects in test, train, val set will be saved
    base_path : object alert directory
        folder where all of the objet, or "Alert" folders are stored
    kernel_path : kernel.pkl
        name of kernel to open or dump gaussian process to
    is_i_filter : True or False
        if you want to include ZTF-filter i or not
    
    '''
    def __init__(self, preprocessed_path, df_bts=None, base_path=None, kernel_path=None, is_i_filter=True):
        self.df_bts = df_bts
        self.base_path = base_path
        self.preprocessed_path = preprocessed_path
        self.is_i_filter = is_i_filter
        self.data = []
        self.data_preprocess = []
        self.kernel = None
        self.kernel_path = kernel_path

    def load_data(self):
        self.data = []
        for file in os.listdir(self.preprocessed_path):
            print(file)
            if file.endswith(".npy"):
                self.data.append(np.load(os.path.join(self.preprocessed_path, file), allow_pickle=True).item())

    def preprocess_data(self, df_bts, base_path, is_prediction=False):
        self.df_bts, self.data_preprocess, self.base_path = df_bts, [], base_path
        for idx, row in tqdm(df_bts.iterrows(), total=df_bts.shape[0], desc="Loading data"):
            try:
                obj_id, target, type_step1 = row['obj_id'], row['type'], row['type_step1']
                #obj_id, target = row['obj_id'], row['type']
                if any(obj_id in file for file in os.listdir(self.preprocessed_path)):
                    continue
                
                photo_df, metadata_df, images = PhotometryProcessor.process_csv(obj_id, df_bts, base_path, is_i_filter=self.is_i_filter), *AlertProcessor.get_process_alerts(obj_id, base_path)  
                
                photo_df, metadata_df = photo_df.sort_values(by='jd'), metadata_df.sort_values(by='jd')
                photo_df = PhotometryProcessor.add_metadata_to_photometry(photo_df, metadata_df, self.is_i_filter)
                photo_df = DataPreprocessor.convert_photometry(photo_df)
                
                max_mjd = min(photo_df['mjd'].max(), 200)
                photo_df = photo_df[photo_df['mjd'] <= max_mjd]
                metadata_df = metadata_df[metadata_df['jd'] <= photo_df['jd'].max()]
   
                metadata_df = DataPreprocessor.preprocess_metadata(metadata_df)
                metadata_df_norm = metadata_df.drop(columns=['jd'])

                start_index = PhotometryProcessor.get_first_valid_index(photo_df)

                if start_index == -1:
                    continue

                if not is_prediction:
                    alert_indices = list(range(len(metadata_df) // 2, len(metadata_df)))
                    if len(alert_indices) > 10:
                        alert_indices = np.round(np.linspace(len(metadata_df) // 2, len(metadata_df) - 1, 10)).astype(int)
                else:
                    alert_indices = list(range(start_index, len(metadata_df)))

                for i in alert_indices:
                    photo_ready = DataPreprocessor.cut_photometry(photo_df, metadata_df, i)
                    if photo_ready is None:
                        break
                    get_index = metadata_df_norm.iloc[i].name
                
                    self.data_preprocess.append({
                            'obj_id': obj_id,
                            'alerte': i,
                            'photometry': photo_ready,
                            'metadata': metadata_df_norm.iloc[i],
                            'images': images[get_index],
                            'target': target,
                            'type_step1': type_step1
                            })    

            except Exception as e:
                print(f"Error processing {obj_id} at index {idx}: {e}")

    def preprocess_and_save(self):
        os.makedirs(self.preprocessed_path, exist_ok=True)
        if self.kernel is None:
            self.load_kernel()

        args = [(sample, self.preprocessed_path, self.kernel, self.is_i_filter) for sample in self.data_preprocess]
        num_workers = multiprocessing.cpu_count() - 1

        with multiprocessing.Pool(num_workers) as pool:
            list(tqdm(pool.imap(DataPreprocessor.process_and_save_sample, args), total=len(self.data_preprocess), desc="Preprocessing"))
    
    def create_kernel(self, kernel_path):
        data = self.data_preprocess.copy()
        random.shuffle(data)
        res_df = pd.concat([sample['photometry'] for sample in data]).reset_index(drop=True).iloc[:20000]
        kernel, _ = gp.fit_2d_gp(res_df, return_kernel=True)
        with open(kernel_path, 'wb') as f:
            pickle.dump(kernel, f)
        self.kernel = kernel

    def load_kernel(self):
        with open(self.kernel_path, 'rb') as f:
            self.kernel = pickle.load(f)

    def __len__(self):
        if len(self.data) == 0:
            return len(self.data_preprocess)
        return len(self.data)


