import joblib
import numpy as np
import pandas as pd
import os
import pickle


def give_me_the_original_ish_alert(train_dataset, train_index, data_df, TRAIN_DATA_PATH):
    
    photometry = train_dataset[train_index][0]
    
    data_match_df = data_df.iloc[train_index]
    alert_name = data_match_df['file']

    alert = np.load(os.path.join(TRAIN_DATA_PATH, alert_name), allow_pickle=True).item()
    
    return alert
