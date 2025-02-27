import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.subplots as sp
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from AppleCider.preprocess.transient_dataset import TransientDataset
from AppleCider.preprocess.data_preprocessor import AlertProcessor, PhotometryProcessor
from AppleCider.preprocess.data_preprocessor import DataPreprocessor, SpectraProcessor
from AppleCider.preprocess.data_preprocessor import DataSorter

from tqdm.auto import tqdm

type_color_dict = { 'SN Ia': 'deepskyblue', 'SN Ia-91T': 'deepskyblue', 'SN Ib/c': 'deepskyblue', 'SN Ia-02cx': 'deepskyblue', 'SN Ia-pec': 'deepskyblue', 'SN Ic': 'deepskyblue', 'SN I': 'deepskyblue','SN Ia-norm': 'deepskyblue', 'SN Ic-BL': 'deepskyblue', 'SN Ibn': 'deepskyblue', 'SN Icn': 'deepskyblue', 'SN Ib-p': 'deepskyblue', 'SN Ia-18byg':'deepskyblue', 'SN Ib': 'deepskyblue', 'SN Ic-norm': 'deepskyblue', 'SN Ibn': 'deepskyblue', 'SN Ib/c': 'deepskyblue', 'SN Ic-SLSN': 'deepskyblue', 'SN Ia-02cx': 'deepskyblue', 'SLSN-I': 'deepskyblue', 'SN Ia-91bg': 'deepskyblue', 'SN Ib-norm': 'deepskyblue', 'SN Ia-18byg': 'deepskyblue', 'SN Ic.5-SLSN': 'deepskyblue', 'SN Ia-CSM': 'deepskyblue', 'SN Ib-pec':'deepskyblue', 'SN Ia-03fg': 'deepskyblue', 'SN II-norm':'deepskyblue', 'SN II': 'lightblue', 'SN IIP':'lightblue', 'SN IIn': 'lightblue', 'SN IIL': 'lightblue', 'SN IIb': 'lightblue', 'SLSN-II': 'lightblue', 'SN II-pec': 'lightblue', 'SN': 'blue', 'SN Ca-rich': 'blue', 'AGN': 'rebeccapurple', 'QSO': 'darkviolet', 'Galactic Nuclei': 'purple', 'Seyfert': 'plum', 'Blazar': 'thistle',  'Stellar variable': 'orange','RR Lyrae': 'orange', 'S Doradus':'salmon', 'Cepheid': 'orange', 'Cataclysmic':'goldenrod', 'Polars': 'goldenrod', 'AM CVn': 'goldenrod', 'Tidal Disruption Event': 'gold', 'Anomolous': 'hotpink', 'U Gem': 'hotpink', 'Classical Nova':'hotpink', 'FU Ori': 'hotpink', 'YSO': 'hotpink', 'long GRB': 'hotpink', 'Pulsar': 'hotpink', 'microlensing': 'hotpink', 'BL Lac': 'hotpink', 'Mira':'hotpink','Nova-like':'hotpink', 'FBOT':'hotpink', 'afterglow': 'hotpink', 'Novae': 'hotpink'}

sdss_fit_color = {'GALAXY':'purple', 'STAR':'yellow', 0:'red'}


# Plot Photometry of object (Magnitude)
def plot_photometry_magnitude(lc, color_dict=None):
    if color_dict is None:
        color_dict = {'ztfi': 'y','ztfg': 'green', 'ztfr': 'red', 'sdssg': 'green', 'sdssr': 'red', 'sdssi': 'y', 'atlasc': 'cyan', 'atlaso': 'orange'}
        color_label = {'ztfi': 'ZTF-i','ztfg': 'ZTF-g', 'ztfr': 'ZTF-r', 'sdssg': 'green', 'sdssr': 'red', 'sdssi': 'y', 'atlasc': 'cyan', 'atlaso': 'orange'}
        line_label = {'ztfi': 'solid','ztfg': 'dashed', 'ztfr': 'dashdot', 'sdssg': 'green', 'sdssr': 'red', 'sdssi': 'y', 'atlasc': 'cyan', 'atlaso': 'orange'}

    flux_bool = False
    
    if 'mag' in lc.columns:
        col_norm = 'mag'
        col_err = 'magerr'
    elif 'flux' in lc.columns:
        col_norm = 'flux'
        col_err = 'flux_error'
        flux_bool = True
    else:
        print("No magnitude or flux column found")
        return
    
    fig, ax1 = plt.subplots(1, 1, figsize=(8,5))
    ymin, ymax = np.inf, -np.inf

    for f in set(lc['filter']):
        tf = lc[lc['filter'] == f]
        
        tf_det = tf[tf[col_norm] >= 3.]
        tf_ul = tf
        if 'snr' in tf.columns:
            tf_ul = tf[tf['snr'] < 3]

        ax1.errorbar(tf_det['mjd'].values,
                     tf_det[col_norm], yerr=tf_det[col_err],
                     #color=color_dict[f], markeredgecolor='k', marker='*',markersize=19 # conference poster size
                     color=color_dict[f], markeredgecolor='k', marker='*',markersize=12,
                     label=color_label[f], linestyle=line_label[f])
        if np.min(tf_det[col_norm]) < ymin:
            ymin = np.min(tf_det[col_norm])
        if np.max(tf_det[col_norm]) > ymax:
            ymax = np.max(tf_det[col_norm])

        if len(tf_ul) != 0:
            if np.min(tf_det[col_norm]) < ymin:
                ymin = np.min(tf_det[col_norm])
            if np.max(tf_det[col_norm]) > ymax:
                ymax = np.max(tf_det[col_norm])
    if flux_bool:
        plt.gca()
        ax1.set_ylabel("Flux", fontsize=12)
    else:
        plt.gca().invert_yaxis()
        ax1.set_ylabel("Magnitude (AB)", fontsize=12)
    ax1.set_xlabel("Time (MJD)", fontsize=12)
    plt.legend(prop={'size': 15}, handlelength=4)
    plt.grid(alpha=0.1)
    ax1.set_title(f"{lc['obj_id'].values[0]} - {lc['type'].values[0]}", fontsize=12, pad=5)
    plt.show()

# Plot Astronomical Images: Science, Reference, Difference
import matplotlib.patches as patches
def plot_image(image, vmin_set, vmax_set):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    titles = ['Science Image', 'Reference Image', 'Difference']

    for i, ax in enumerate(axes):
        ax.imshow(image[:, :, i], cmap='viridis_r', vmin=vmin_set, vmax=vmax_set)
        ax.set_title(titles[i], fontsize=14)
        # ADD BOX, IT WORKS:
        rect = patches.Rectangle((24, 24), 15, 15, linewidth=3, edgecolor='y', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.axis('off')
    plt.show()  

    
    
    
    
class plot_images():    

    def plot_image_pres(image):
        """ use to graph images of objects after basic preprocessing"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.subplots_adjust(wspace=0.01, hspace=0.1)
        titles = ['Science Image', 'Reference Image', 'Difference']
    
        for i, ax in enumerate(axes):
            ax.imshow(image[:, :, i], cmap='viridis_r', vmin=0.0, vmax=.0999 )
            ax.set_title(titles[i], fontsize=14)
            # ADD BOX, IT WORKS:
            rect = patches.Rectangle((24, 24), 15, 15, linewidth=3, edgecolor='y', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.axis('off')
        plt.show() 
        
        
    def plot_image_tensor(image):
        """ use to graph images of objects from dataloader"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.subplots_adjust(wspace=0.01, hspace=0.1)
        titles = ['Science Image', 'Reference Image', 'Difference']
        
        print("image: \n",image)
    
    
        for i, ax in enumerate(axes):
            ax.imshow(image[i, :, :], cmap='viridis_r', vmin=0.0, vmax=.0999 )
            ax.set_title(titles[i], fontsize=14)
            # ADD BOX, IT WORKS:
            rect = patches.Rectangle((24, 24), 15, 15, linewidth=3, edgecolor='y', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.axis('off')
        plt.show() 


    
def mass_plot_photometry(obj_id_list, SEDM_dataset, data_dir, max_mjd=10,color_dict=None):
    
    for obj_id in obj_id_list:
        # get info from alerts    
        photo_df = PhotometryProcessor.process_csv(obj_id, SEDM_dataset, data_dir)
        metadata_df, images = AlertProcessor.get_process_alerts(obj_id, data_dir)
        # do something else i guess
        photo_df, metadata_df = photo_df.sort_values(by='jd'), metadata_df.sort_values(by='jd')
        photo_df = PhotometryProcessor.add_metadata_to_photometry(photo_df, metadata_df)
        # convert magnitude to flux, normalize mjd
        photo_df = DataPreprocessor.convert_photometry(photo_df)
        
        photo_df = photo_df[photo_df['mjd'] <= max_mjd]
            
        if color_dict is None:
            color_dict = {'ztfi': 'y','ztfg': 'green', 'ztfr': 'red', 'sdssg': 'green', 'sdssr': 'red', 'sdssi': 'y', 'atlasc': 'cyan', 'atlaso': 'orange'}
            color_label = {'ztfi': 'ZTF-i','ztfg': 'ZTF-g', 'ztfr': 'ZTF-r', 'sdssg': 'green', 'sdssr': 'red', 'sdssi': 'y', 'atlasc': 'cyan', 'atlaso': 'orange'}
            line_label = {'ztfi': 'solid','ztfg': 'dashed', 'ztfr': 'dashdot', 'sdssg': 'green', 'sdssr': 'red', 'sdssi': 'y', 'atlasc': 'cyan', 'atlaso': 'orange'}
    
        if 'flux' in photo_df.columns:
            col_norm = 'flux'
            col_err = 'flux_error'
            flux_bool = True
        else:
            print("No flux column found")
            return
        
        fig, ax1 = plt.subplots(1, 1, figsize=(8,5))
        ymin, ymax = np.inf, -np.inf
    
        for f in set(photo_df['filter']):
            tf = photo_df[photo_df['filter'] == f]
            
            tf_det = tf[tf[col_norm] >= 3.]
            tf_ul = tf
            if 'snr' in tf.columns:
                tf_ul = tf[tf['snr'] < 3]
    
            ax1.errorbar(tf_det['mjd'].values, tf_det[col_norm], yerr=tf_det[col_err], color=color_dict[f], markeredgecolor='k',
                         marker='.',markersize=10, alpha=0.6,label=color_label[f], linestyle=line_label[f])
            if np.min(tf_det[col_norm]) < ymin:
                ymin = np.min(tf_det[col_norm])
            if np.max(tf_det[col_norm]) > ymax:
                ymax = np.max(tf_det[col_norm])
    
            if len(tf_ul) != 0:
                if np.min(tf_det[col_norm]) < ymin:
                    ymin = np.min(tf_det[col_norm])
                if np.max(tf_det[col_norm]) > ymax:
                    ymax = np.max(tf_det[col_norm])
                    
        plt.gca()
        ax1.set_ylabel("Flux", fontsize=12)
        ax1.set_xlim(-0.5,11)
        ax1.set_xlabel("Time (MJD)", fontsize=12)
        
        plt.legend(prop={'size': 14}, handlelength=4)
        plt.grid(alpha=0.1)
        ax1.set_title(f"{photo_df['obj_id'].values[0]} - {photo_df['type'].values[0]}", fontsize=12, pad=5)
        plt.show()
    
    
    
class plot_dataset:


    def plot_dataset_item(dataset, index, data_df):
        """
        Plots photometry, images, and spectra for a specific dataset item in a single row.
    
        Parameters:
        dataset: The dataset object containing data items.
        item_id (int): The ID of the dataset item to plot.
        """
        
        data_match_df = data_df.iloc[index]
        obj_id = data_match_df['name']
        obj_label = data_match_df['type']
        obj_alert = data_match_df['file']
        
        photometry, photometry_mask, metadata, images, spectra, label = dataset[index]
        
        print(f"{obj_id}: {obj_alert}")
        print("metadata columns: \n sgscore1, sgscore2, distpsnr1, distpsnr2, ra, dec, nmtchps, sharpnr, scorr, sky \n", metadata)
        
        fig = plt.figure(figsize=(24, 6))
        fig.suptitle(f'{obj_id}, {obj_label}: {obj_alert}', y=1.02, fontsize=20)
        
        gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1.5, 1.5])
    
        ax_img1 = fig.add_subplot(gs[0, 0])
        ax_img1.imshow(images[0, :, :], cmap='viridis_r')
        ax_img1.axis('off')
        ax_img1.set_title('Science Image')
    
        ax_img2 = fig.add_subplot(gs[0, 1])
        ax_img2.imshow(images[1, :, :], cmap='viridis_r')
        ax_img2.axis('off')
        ax_img2.set_title('Reference Image')
    
        ax_img3 = fig.add_subplot(gs[0, 2])
        ax_img3.imshow(images[2, :, :], cmap='viridis_r')
        ax_img3.axis('off')
        ax_img3.set_title('Difference')
        
        
        ## photometry from train_dataset[#]
        dataset_dates = photometry[:,0] ; dataset_ztfg = photometry[:,1]
        dataset_ztfr = photometry[:,2] ; dataset_ztfi = photometry[:,3]
    
        ax_photometry = fig.add_subplot(gs[0, 3])
        ax_photometry.scatter(dataset_dates,dataset_ztfg ,label='ztf-g',marker='v',alpha=0.75,c='green',edgecolor='black',s=200)
        ax_photometry.scatter(dataset_dates,dataset_ztfr ,label='ztf-r',marker='^',alpha=0.75,c='red',edgecolor='black',s=200)
        ax_photometry.scatter(dataset_dates,dataset_ztfi, label='ztf-i',marker='>',alpha=0.75,c='yellow',edgecolor='black',s=200)
        ax_photometry.set_title('Photometry')
        ax_photometry.grid(alpha=0.15)
        #ax_photometry.legend(prop={'size': 12}, handlelength=5)
        ax_photometry.legend()
    
        ax_spectra = fig.add_subplot(gs[0, 4])
        ax_spectra.plot(spectra[0], label='Spectra', color='yellowgreen')
        ax_spectra.set_title('Spectra')
        ax_spectra.grid(alpha=0.15)
        ax_spectra.legend()
    
        plt.tight_layout()
        plt.show()
        
    def compare_dataset_photometry(photometry, obj_id, SEDM_dataset, data_dir):
        
        ''' basically just check to that an object's original photometry
            matches with the photometry tensor from the dataset generator '''
        
        ## photometry from train_dataset[#]
        dataset_dates = photometry[:,0]
        dataset_ztfg = photometry[:,1]    ## ztf-g
        dataset_ztfr = photometry[:,2]    ## ztf-r
        dataset_ztfi = photometry[:,3]    ## ztf-i
        
        obj_df = SEDM_dataset[SEDM_dataset['obj_id'] == obj_id]
        obj_label = obj_df['type'].values[0]
        
        ## now, get the object's pre-processed photometry, process it a little bit to get flux
        photo_df = PhotometryProcessor.process_csv(obj_id, SEDM_dataset, data_dir)
        metadata_df, images = AlertProcessor.get_process_alerts(obj_id, data_dir)
        photo_df, metadata_df = photo_df.sort_values(by='jd'), metadata_df.sort_values(by='jd')
        photo_df = PhotometryProcessor.add_metadata_to_photometry(photo_df, metadata_df)
        ## convert magnitude to flux, normalize mjd
        photo_df = DataPreprocessor.convert_photometry(photo_df)
        photo_df = photo_df[photo_df['mjd'] <= 10]     ## only want photometry by first 10 alerts
        
        ## for plotting:
        color_dict = {'ztfi': 'y','ztfg': 'green', 'ztfr': 'red', 'sdssg': 'green', 'sdssr': 'red', 'sdssi': 'y', 'atlasc': 'cyan', 'atlaso': 'orange'}
        color_label = {'ztfi': 'ZTF-i','ztfg': 'ZTF-g', 'ztfr': 'ZTF-r', 'sdssg': 'green', 'sdssr': 'red', 'sdssi': 'y', 'atlasc': 'cyan', 'atlaso': 'orange'}
        line_label = {'ztfi': 'solid','ztfg': 'dashed', 'ztfr': 'dashdot', 'sdssg': 'green', 'sdssr': 'red', 'sdssi': 'y', 'atlasc': 'cyan', 'atlaso': 'orange'}
        
        if 'flux' in photo_df.columns:
            col_norm = 'flux'
            col_err = 'flux_error'
            flux_bool = True
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
        fig.tight_layout()
        
        fig.suptitle(f'{obj_id}: {obj_label}', y=1.12, fontsize=14)
        plt.subplots_adjust(wspace=0.25, hspace=2.4)
        ymin, ymax = np.inf, -np.inf
        
        for f in set(photo_df['filter']):
            tf = photo_df[photo_df['filter'] == f]
            
            tf_det = tf[tf[col_norm] >= 3.]
            tf_ul = tf
            if 'snr' in tf.columns:
                tf_ul = tf[tf['snr'] < 3]
        
            ## slightly processed photometry
            ax1.errorbar(tf_det['mjd'].values, tf_det[col_norm], yerr=tf_det[col_err], color=color_dict[f], markeredgecolor='k',
                         marker='*',markersize=20, alpha=0.6,label=color_label[f], linestyle=line_label[f])
            if np.min(tf_det[col_norm]) < ymin:
                ymin = np.min(tf_det[col_norm])
            if np.max(tf_det[col_norm]) > ymax:
                ymax = np.max(tf_det[col_norm])
        
            if len(tf_ul) != 0:
                if np.min(tf_det[col_norm]) < ymin:
                    ymin = np.min(tf_det[col_norm])
                if np.max(tf_det[col_norm]) > ymax:
                    ymax = np.max(tf_det[col_norm])
                    
        ## data photometry
        ax2.scatter(dataset_dates,dataset_ztfg, marker='v',alpha=0.75,c='green', s=120, edgecolor='black', label='ztf-g')
        ax2.scatter(dataset_dates,dataset_ztfr, marker='^',alpha=0.75,c='r', s=120, edgecolor='black', label='ztf-r')
        ax2.scatter(dataset_dates,dataset_ztfi, marker='>',alpha=0.75,c='y', s=120,edgecolor='black', label='ztf-i')
        
        ax2.set_ylabel("Normalized Flux", fontsize=12) ; ax2.set_xlabel("Time (MJD)", fontsize=12)
        ax2.set_title('Dataset Photometry')
        
        ax1.set_ylabel("Flux", fontsize=12) ; ax1.set_xlabel("Time (MJD)", fontsize=12)
        ax1.set_title('Photometry')
        
        ax1.grid(alpha=0.1) ; ax2.grid(alpha=0.1)
        ax1.legend(prop={'size': 10}, handlelength=4) ; ax2.legend(prop={'size': 10}, handlelength=4)
        plt.show()
        
        
    
    
    
    
    
def plot_spectra(obj_id, obj_type_df, data_dir):
        
    obj_df = obj_type_df[obj_type_df['obj_id'] == f'{obj_id}']    
    obj_class = obj_df['type'].values[0]
    
    spectra_df = SpectraProcessor.get_spectra_df(obj_id, data_dir)
    instrument_name = spectra_df['instrument_name'][0]

    fig, (ax1) = plt.subplots(1, 1)
    #plt.rcParams["figure.figsize"] = (9,2)
    fig.suptitle(f'{obj_id}')
    # fritz spectra
    ax1.plot(spectra_df['wavelength'], spectra_df['flux'], color=type_color_dict[obj_class])
    ax1.set_title(f'{instrument_name}: {obj_class}')
    plt.show()

# Plot Gaussian process for object's Light Curve
#def plot_gp(raw_df, gp_df, type_obj=None, alerte=None):
#    color_dict = {'ztfg': 'rgba(0, 128, 0, 0.2)', 'ztfr': 'rgba(255, 0, 0, 0.2)', 'ztfi': 'rgba(255, 255, 0, 0.2)'}
#    line_color_dict = {'ztfg': 'rgba(0, 128, 0, 1.0)', 'ztfr': 'rgba(255, 0, 0, 1.0)', 'ztfi': 'rgba(255, 255, 0, 1.0)'}
#    star_colors = {'ztfg': 'green', 'ztfr': 'red', 'ztfi': 'rgb(253, 218, 13)'}
#    color_label = {'ztfg': 'ZTF-g', 'ztfr': 'ZTF-r', 'ztfi': 'ZTF-i'}
#    line_label = {'ztfi': 'solid','ztfg': 'dash', 'ztfr': 'dashdot', 'sdssg': 'green', 'sdssr': 'red', 'sdssi': 'y', 'atlasc': 'cyan', 'atlaso': 'orange',}
#    
#    fig = go.Figure()
#
#    for filter_name in ['ztfg', 'ztfr', 'ztfi']:
#        flux_col = f'flux_{filter_name}'
#        error_col = f'flux_error_{filter_name}'
#        fig.add_trace(go.Scatter(
#            x=gp_df['mjd'], y=gp_df[flux_col],
#            mode='lines', name=f'GP {color_label[filter_name]}',
#            line=dict(color=line_color_dict[filter_name], dash=line_label[filter_name], width=2)
#        ))
#        fig.add_trace(go.Scatter(
#            x=gp_df['mjd'], y=gp_df[flux_col] + gp_df[error_col],
#            mode='lines', line=dict(color=line_color_dict[filter_name].replace('1.0', '0')),
#            showlegend=False, name=f'Upper Bound {color_label[filter_name]}'
#        ))
#        fig.add_trace(go.Scatter(
#            x=gp_df['mjd'],y=gp_df[flux_col] - gp_df[error_col],
#            mode='lines', line=dict(color=line_color_dict[filter_name].replace('1.0', '0')),
#            fill='tonexty', fillcolor=color_dict[filter_name],
#            showlegend=False, name=f'Lower Bound {color_label[filter_name]}'
#        ))
#
#    for filter_name in raw_df['filter'].unique():
#        filter_data = raw_df[raw_df['filter'] == filter_name]
#        fig.add_trace(go.Scatter(
#            x=filter_data['mjd'], y=filter_data['flux'],
#            mode='markers', name=f'Raw {color_label[filter_name]}',
#            marker=dict(color=line_color_dict[filter_name], size=7.5, opacity=0.99,line=dict(width=0.8, color="Black")), marker_symbol = 'star', 
#            hovertemplate=(
#                'mjd: %{x}<br>'
#                'flux: %{y}<br>'
#                'filter: %{text}<br>'
#                'obj_id: ' + filter_data['obj_id'].iloc[0] + '<br>'
#            ),
#            text=filter_data['filter'],
#        ))
#    fig.update_layout(
#        xaxis_title='Time (Days)',
#        yaxis_title='Flux',
#        width=600, height=300, 
#        font = dict(size=9)
#    )
#    fig.update_xaxes(showline=True, linewidth=1, linecolor='black') ; fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
#    title = f'Light Curve: {raw_df["obj_id"].iloc[0]}'
#
#    if type_obj is not None:
#        title += f' - Type: {type_obj}'
#
#    if alerte is not None:
#        title += f' - Alert: {alerte}'
#    fig.update_layout(title=title, width=600, height=400)
#    fig.show()

# Plot Photometry
#def plot_photometry(obj_id, alerte=None, type_obj=None):
#    df_bts = pd.read_csv('obj_type_steps.csv')
#    base_path = '(aj)data_all/'
#    photo_df, metadata_df, _ = PhotometryProcessor.process_csv(obj_id, df_bts, base_path), *AlertProcessor.get_process_alerts(obj_id, base_path)
#    photo_df, metadata_df = photo_df.sort_values(by='jd'), metadata_df.sort_values(by='jd')
#    photo_df = PhotometryProcessor.add_metadata_to_photometry(photo_df, metadata_df)
#    photo_df = DataPreprocessor.convert_photometry(photo_df)
#
#    max_mjd = min(photo_df['mjd'].max(), 90)
#    photo_ready = photo_df[photo_df['mjd'] <= max_mjd]
#    metadata_df = metadata_df[metadata_df['jd'] <= photo_ready['jd'].max()]
#
#
#    kernel = pickle.load(open('kernel.pkl', 'rb'))
#    gp_final = gp.process_gaussian(photo_ready, kernel=kernel, number_gp=200)
#
#    for i, jd in enumerate(metadata_df['jd'], start=1):
#        photo_ready.loc[photo_ready['jd'] == jd, 'alert_num'] = i
#
#    if 'flux_ztfi' not in gp_final.columns:
#        gp_final['flux_ztfi'] = 0
#        gp_final['flux_error_ztfi'] = 0
#
#    if 'flux_ztfg' not in gp_final.columns:
#        gp_final['flux_ztfg'] = 0
#        gp_final['flux_error_ztfg'] = 0
#
#    if 'flux_ztfr' not in gp_final.columns:
#        gp_final['flux_ztfr'] = 0
#        gp_final['flux_error_ztfr'] = 0  
#
#    return photo_ready, gp_final
#


# Plot Model Accuracy & Losses
#def plot_history(history):
#    if not isinstance(history, dict):
#        history = history.history
#    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('Model accuracy', 'Model loss'), shared_xaxes=True)
#    # Accuracy
#    fig.add_trace(go.Scatter(y=history['accuracy'], mode='lines', name='Train Accuracy'), row=1, col=1)
#    fig.add_trace(go.Scatter(y=history['val_accuracy'], mode='lines', name='Validation Accuracy'), row=1, col=1)
#    # Losses
#    fig.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Train Loss'), row=1, col=2)
#    fig.add_trace(go.Scatter(y=history['val_loss'], mode='lines', name='Validation Loss'), row=1, col=2)
#    fig.update_xaxes(title_text='Epoch', row=1, col=1)
#    fig.update_xaxes(title_text='Epoch', row=1, col=2)
#    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
#    fig.update_yaxes(title_text='Loss', row=1, col=2)
#    fig.update_layout(title='Training History', height=400, width=700, showlegend=True)
#    fig.show()
#
## Plot ROC Curve
#def plot_roc(y_true, y_pred):
#    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
#    roc_auc = auc(fpr, tpr)
#    plt.figure(figsize=(8, 6))
#    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
#    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#    plt.xlim([0.0, 1.0]) ; plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate') ; plt.ylabel('True Positive Rate')
#    plt.title('Receiver Operating Characteristic (ROC)')
#    plt.legend(loc="lower right")
#    plt.show()
#
## Plot Confusion Matrix
#def plot_confusion_matrix(y_true, y_pred_max, labels):
#    cm = confusion_matrix(y_true, y_pred_max)
#    plt.figure(figsize=(10, 8))
#    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Not SN', 'SN'], yticklabels=['Not SN', 'SN'])
#    plt.xticks(rotation=0) ; plt.yticks(rotation=0)
#    plt.xlabel('Predicted Type', labelpad=30, fontsize=12)
#    plt.ylabel('Actual Type', labelpad=28, fontsize=12)
#    plt.title('Confusion Matrix', fontsize=14, pad=10)
#    plt.show()

# Plot Confusion Matrix, ROC together:
def plot_pred_set(y_true, y_pred_max, labels):
    # ROC:
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]) ; plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate') ; plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    # CONFUSION:
    cm = confusion_matrix(y_true, y_pred_max)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Not SN', 'SN'], yticklabels=['Not SN', 'SN'])
    plt.xticks(rotation=0) ; plt.yticks(rotation=0)
    plt.xlabel('Predicted Type', labelpad=30, fontsize=12)
    plt.ylabel('Actual Type', labelpad=28, fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, pad=10)