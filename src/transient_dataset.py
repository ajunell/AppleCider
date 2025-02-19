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
    
    def __init__(self, preprocessed_path, df_bts=None, base_path=None):
        
        self.preprocessed_path = preprocessed_path
        self.df_bts = df_bts
        self.base_path = base_path
        self.data = []
        self.data_preprocess = []
        #self.normalize_photometry = normalize_photometry
        #self.include_spectra = include_spectra

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
                
                ## decide "alerts" to sample from 
                alert_indices = list(range(len(metadata_df) // 2, len(metadata_df)))
          
                for i in alert_indices:
                    photo_ready = DataPreprocessor.cut_photometry(photo_df, metadata_df, i)
                    if photo_ready is None:
                        break
                    ## get matching index for metadata, image    
                    get_index = metadata_df_norm.iloc[i].name
                   
                    self.data_preprocess.append({
                            'obj_id': obj_id,
                            'alerte': i,
                            'photometry': photo_ready,
                            'metadata': metadata_df_norm.iloc[i],
                            'images': images[get_index],
                            'spectra': spectra,
                            'target': target, })
                        

            except Exception as e:
                print(f"Error processing {obj_id} at index {idx}: {e}")

                 
    def process_and_save_sample(args):
        ''' save dictionary w/processed photometry, metadata, images to .npy at desired path '''
        
        res_dict = {}
        
        sample, save_dir = args
        obj_id = sample['obj_id']
        alerte = sample['alerte']    ## keep it in french
        type_obj = sample['target']
        
        photometry = sample['photometry']
        if len(photometry) == 0:
            return

        save_path = os.path.join(save_dir, f"{obj_id}_alert_{alerte}.npy")
        if os.path.exists(save_path):
            return

        res_df = pd.DataFrame()
        
        photometry = sample['photometry'].pivot_table(index=['mjd'], columns='filter', values=['flux', 'flux_error'])
        photometry = photometry.reset_index()
        photometry.columns = [col[0] if col[0] == 'mjd' else '_'.join(col).strip() for col in photometry.columns.values]
        photometry['obj_id'] = obj_id

        res_df = pd.concat([res_df, photometry])
        res_df = res_df.reset_index(drop=True, inplace=True)
              
        ## if you want flux error sometime, don't forget to add back in to column list
        columns = ['flux_ztfg', 'flux_ztfr', 'flux_ztfi']
        for col in columns:
            if col not in photometry.columns:
                photometry[col] = 0.
        
        ## if you want flux error sometime, don't forget to add back in to column list
        photometry = photometry[['obj_id', 'mjd', 'flux_ztfg', 'flux_ztfr', 'flux_ztfi']]
        photometry = photometry.fillna(0)
        
        photometry = PhotometryProcessor.normalize_light_curve(photometry)

        ## keep only mjd, flux ztfr, flux ztfg, flux_ztfi
        useful_columns = ['mjd', 'flux_ztfg', 'flux_ztfr', 'flux_ztfi']
        photometry = photometry[useful_columns].values
        
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
        args = [(sample, self.preprocessed_path) for sample in self.data_preprocess]
 
        [TransientDataset.process_and_save_sample(args) for args in tqdm(args, desc="Saving Preprocessed Alerts", leave=True)]