import os
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

from src.data_preprocessor import AlertProcessor
from src.data_preprocessor import PhotometryProcessor
from src.data_preprocessor import SpectraProcessor
from src.data_preprocessor import DataPreprocessor

from torch.nn.utils.rnn import pad_sequence
            
            
class TransientDataset():
    
    def __init__(self, preprocessed_path, df_bts=None, base_path=None, normalize_photometry=True):
        
        self.preprocessed_path = preprocessed_path
        self.df_bts = df_bts
        self.base_path = base_path
        self.data = []
        self.data_preprocess = []
        self.normalize_photometry = normalize_photometry

    def preprocess_data(self, df_bts, base_path):
        ''' preprocess photometry, metadata, images  by creating dictionary for each object alert sample'''
        
        self.df_bts, self.data_preprocess, self.base_path = df_bts, [], base_path
                 
        for idx, row in tqdm(df_bts.iterrows(), total=df_bts.shape[0], desc="Loading data", leave=True):
            try:
                obj_id, target = row['obj_id'], row['type']
                if any(obj_id in file for file in os.listdir(self.preprocessed_path)):
                    continue
                ## get photometry, metadata, images
                photo_df, metadata_df, images = PhotometryProcessor.process_csv(obj_id, df_bts, base_path), *AlertProcessor.get_process_alerts(obj_id, base_path)
                ## fix photometry
                photo_df, metadata_df = photo_df.sort_values(by='jd'), metadata_df.sort_values(by='jd')
                photo_df = PhotometryProcessor.add_metadata_to_photometry(photo_df, metadata_df)
                ## convert magnitude to flux, get flux error
                photo_df = DataPreprocessor.convert_photometry(photo_df)

                max_mjd = min(photo_df['mjd'].max(), 10)
                photo_df = photo_df[photo_df['mjd'] <= max_mjd]
                metadata_df = metadata_df[metadata_df['jd'] <= photo_df['jd'].max()]

                metadata_df = DataPreprocessor.preprocess_metadata(metadata_df)
                metadata_df_norm = metadata_df.drop(columns=['jd'])

                ## get wavelength, flux from spectra.csv 
                spectra = SpectraProcessor.read_spectra_csv(obj_id, base_path)
                spectra = SpectraProcessor.preprocess_spectra(spectra)

                ## find first valid photometry index
                start_index = PhotometryProcessor.get_first_valid_index(photo_df)
                if start_index == -1:
                    continue
                
                alert_indices = list(range(len(metadata_df) // 2, len(metadata_df)))
                if len(alert_indices) > 10:
                    alert_indices = np.round(np.linspace(len(metadata_df) // 2, len(metadata_df) - 1, 10)).astype(int)
                
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
                            'spectra': spectra,
                            'target': target,
                    })
            except Exception as e:
                print(f"Error processing {obj_id} at index {idx}: {e}")

                 
    def process_and_save_sample(args):
        ''' save dictionary w/processed photometry, metadata, images to .npy at desired path '''
        
        res_dict = {}
        
        sample, save_dir, normalize_photometry = args
        obj_id = sample['obj_id']
        ## keeping it in french
        alerte = sample['alerte']
        type_obj = sample['target']

        save_path = os.path.join(save_dir, f"{obj_id}_alert_{alerte}.npy")
        if os.path.exists(save_path):
            return
        
        photometry = sample['photometry']
        ## remove filters with 0 points
        photometry = PhotometryProcessor.remove_filter(photometry)
        
        if len(photometry) == 0:
            return

        res_df = pd.DataFrame()
        
        last_mjd = sample['photometry']['mjd'].max()
        sample['photometry'].loc[sample['photometry']['mjd'] > last_mjd, ['flux', 'flux_error']] = 0
        photometry = sample['photometry'].pivot_table(index=['mjd'], columns='filter', values=['flux', 'flux_error'])
        photometry = photometry.reset_index()
        photometry.columns = [col[0] if col[0] == 'mjd' else '_'.join(col).strip() for col in photometry.columns.values]
        photometry['obj_id'] = obj_id

        res_df = pd.concat([res_df, photometry])
        res_df = res_df.reset_index(drop=True, inplace=True)
              
        columns = ['flux_ztfg', 'flux_error_ztfg', 'flux_ztfr', 'flux_error_ztfr']
        
        for col in columns:
            if col not in photometry.columns:
                photometry[col] = 0.
         
        ## normalize photometry   
        if normalize_photometry:
            photometry = PhotometryProcessor.normalize_light_curve(photometry)

        ## get date, flux ztfr, flux ztg
        useful_columns = ['mjd', 'flux_ztfg', 'flux_ztfr']
        photometry = photometry[useful_columns].values
        # remove mjd, only ztfg, ztfr
        photometry = photometry[:, 1:]
        
        ## replace nan with zero
        photometry[np.isnan(photometry)] = 0
        
        res_dict.update({
            'obj_id': obj_id,
            'photometry': photometry,
            'metadata': sample['metadata'],
            'images': sample['images'],
            'spectra':sample['spectra'],
            'target': sample['target'],
            'alerte': alerte})

        np.save(save_path, res_dict)
            
    def preprocess_and_save(self):
        os.makedirs(self.preprocessed_path, exist_ok=True)
        args = [(sample, self.preprocessed_path, self.normalize_photometry) for sample in self.data_preprocess]
 
        [TransientDataset.process_and_save_sample(args) for args in tqdm(args, desc="Processing Objects", leave=True)]
    
    