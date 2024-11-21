import os
import pandas as pd
from tqdm import tqdm
import pickle
import multiprocessing
import numpy as np
import random

import src_dataloader.gaussian_process as gp
from src_dataloader.data_preprocessor import AlertProcessor
from src_dataloader.data_preprocessor import PhotometryProcessor
from src_dataloader.data_preprocessor import DataPreprocessor

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
                if any(obj_id in file for file in os.listdir(self.preprocessed_path)):
                    continue
                photo_df, metadata_df, images = PhotometryProcessor.process_csv(obj_id, df_bts, base_path, is_i_filter=self.is_i_filter), *AlertProcessor.get_process_alerts(obj_id, base_path)
                
                photo_df, metadata_df = photo_df.sort_values(by='jd'), metadata_df.sort_values(by='jd')
                photo_df = PhotometryProcessor.add_metadata_to_photometry(photo_df, metadata_df, self.is_i_filter)
                # convert magnitude to flux, jd to MJD
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
                            'type_step1': type_step1,

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