import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from tqdm.auto import tqdm
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

## 
type_color_dict = { 'SN Ia': 'deepskyblue', 'SN Ia-91T': 'deepskyblue', 'SN Ib/c': 'deepskyblue', 'SN Ia-02cx': 'deepskyblue', 'SN Ia-pec': 'deepskyblue', 'SN Ic': 'deepskyblue', 'SN I': 'deepskyblue','SN Ia-norm': 'deepskyblue', 'SN Ic-BL': 'deepskyblue', 'SN Ibn': 'deepskyblue', 'SN Icn': 'deepskyblue', 'SN Ib-p': 'deepskyblue', 'SN Ia-18byg':'deepskyblue', 'SN Ib': 'deepskyblue', 'SN Ic-norm': 'deepskyblue', 'SN Ibn': 'deepskyblue', 'SN Ib/c': 'deepskyblue', 'SN Ic-SLSN': 'deepskyblue', 'SN Ia-02cx': 'deepskyblue', 'SLSN-I': 'deepskyblue', 'SN Ia-91bg': 'deepskyblue', 'SN Ib-norm': 'deepskyblue', 'SN Ia-18byg': 'deepskyblue', 'SN Ic.5-SLSN': 'deepskyblue', 'SN Ia-CSM': 'deepskyblue', 'SN Ib-pec':'deepskyblue', 'SN Ia-03fg': 'deepskyblue','SN II-norm':'deepskyblue', 'SN II': 'lightblue', 'SN IIP':'lightblue', 'SN IIn': 'lightblue', 'SN IIL': 'lightblue', 'SN IIb': 'lightblue', 'SLSN-II': 'lightblue', 'SN II-pec': 'lightblue', 'SN': 'blue', 'SN Ca-rich': 'blue', 'AGN': 'rebeccapurple', 'QSO': 'darkviolet', 'Galactic Nuclei': 'purple', 'Seyfert': 'plum', 'Blazar': 'thistle',  'Stellar variable': 'orange','RR Lyrae': 'orange', 'S Doradus':'salmon', 'Cepheid': 'orange', 'Cataclysmic':'goldenrod', 'Polars': 'goldenrod', 'AM CVn': 'goldenrod', 'Tidal Disruption Event': 'gold', 'Anomolous': 'hotpink', 'U Gem': 'hotpink', 'Classical Nova':'hotpink', 'FU Ori': 'hotpink', 'YSO': 'hotpink', 'long GRB': 'hotpink', 'Pulsar': 'hotpink', 'microlensing': 'hotpink', 'BL Lac': 'hotpink', 'Mira':'hotpink','Nova-like':'hotpink', 'FBOT':'hotpink', 'afterglow': 'hotpink', 'Novae': 'hotpink'}


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
    
    
def mass_plot_photometry(obj_id_list, SEDM_dataset, data_dir, max_mjd=10,color_dict=None):
    
    for obj_id in obj_id_list:
        ## get info from alerts    
        photo_df = PhotometryProcessor.process_csv(obj_id, SEDM_dataset, data_dir)
        metadata_df, images = AlertProcessor.get_process_alerts(obj_id, data_dir)
        ## do something else i guess
        photo_df, metadata_df = photo_df.sort_values(by='jd'), metadata_df.sort_values(by='jd')
        photo_df = PhotometryProcessor.add_metadata_to_photometry(photo_df, metadata_df)
        ## convert magnitude to flux, normalize mjd
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


## Plot Astronomical Images: Science, Reference, Difference
import matplotlib.patches as patches
def plot_image(image, vmin_set, vmax_set):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    titles = ['Science Image', 'Reference Image', 'Difference']

    for i, ax in enumerate(axes):
        ax.imshow(image[:, :, i], cmap='viridis_r', vmin=vmin_set, vmax=vmax_set)
        ax.set_title(titles[i], fontsize=14)
        ## box
        rect = patches.Rectangle((24, 24), 15, 15, linewidth=3, edgecolor='y', facecolor='none')
        ## Add the patch to the Axes
        ax.add_patch(rect)
        ax.axis('off')
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

    
    
    
## image plots that are actually in use in the notebooks    
class plot_images():    

    def plot_image_pres(image):
        """ use to graph images of objects after basic preprocessing"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.subplots_adjust(wspace=0.01, hspace=0.1)
        titles = ['Science Image', 'Reference Image', 'Difference']
    
        for i, ax in enumerate(axes):
            ax.imshow(image[:, :, i], cmap='viridis_r', vmin=0.0, vmax=.0999 )
            ax.set_title(titles[i], fontsize=14)
            ## box
            rect = patches.Rectangle((24, 24), 15, 15, linewidth=3, edgecolor='y', facecolor='none')
            ## Add the patch to the Axes
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
            ## box
            rect = patches.Rectangle((24, 24), 15, 15, linewidth=3, edgecolor='y', facecolor='none')
            ## Add the patch to the Axes
            ax.add_patch(rect)
            ax.axis('off')
        plt.show() 

        
class plot_dataset:


    def plot_dataset_item(dataset, dataset_index):
        """
        Plots photometry, images, and spectra for a specific dataset item in a single row.
        """
        
        photometry, metadata, images, spectra, target = dataset[dataset_index]
        
        class_map = {'SN Ia':0 ,'SN Ic':1,  'SN Ib':2 , 'SN II': 3, 'SN IIP': 4, 'SN IIn': 5,
                 'SN IIb': 6, 'Cataclysmic': 7, 'AGN': 8, 'Tidal Disruption Event': 9}
        obj_label = [key for key, value in class_map.items() if value == target][0]
        
        print("metadata columns: \n sgscore1, sgscore2, distpsnr1, distpsnr2, ra, dec, nmtchps, sharpnr, scorr, sky \n", metadata)
        
        fig = plt.figure(figsize=(24, 6))
        fig.suptitle(f'dataset: {obj_label}', y=1.02, fontsize=20)
    
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
        
        
    def plot_dataset_item_named(dataset, dataset_index,train_files):
        """
        Plots photometry, images, and spectra for a specific dataset item in a single row.
    
        Parameters:
        dataset: The dataset object containing data items.
        item_id (int): The ID of the dataset item to plot.
        """
    
        train_files_sort = sorted(train_files)
        
        obj_alert = train_files_sort[dataset_index]    ## obj alert file
        obj_id = obj_alert[:12]                        ## gets object id from alert file 
        ## obj type/class from alert
        class_map = {'SN Ia':0 ,'SN Ic':1,  'SN Ib':2 , 'SN II': 3, 'SN IIP': 4, 'SN IIn': 5,
                     'SN IIb': 6, 'Cataclysmic': 7, 'AGN': 8, 'Tidal Disruption Event': 9}
        obj_label = [key for key, value in class_map.items() if value == target][0]
        
        print(f"{obj_id}: {obj_alert}")
        photometry, metadata, images, spectra, target = dataset[dataset_index]
        ## photometry
        dataset_dates = photometry[:,0] ; dataset_ztfg = photometry[:,1]
        dataset_ztfr = photometry[:,2] ; dataset_ztfi = photometry[:,3]
        ## metadata
        print("metadata columns: \n sgscore1, sgscore2, distpsnr1, distpsnr2, ra, dec, nmtchps, sharpnr, scorr, sky \n", metadata)
        
        fig = plt.figure(figsize=(24, 6))
        fig.suptitle(f'{obj_id}, {obj_label}: {obj_alert}', y=1.02, fontsize=20)
        
        gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1.5, 1.5])
        
        ## science image, reference image, difference 
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
        
        ## photometry
        ax_photometry = fig.add_subplot(gs[0, 3])
        ax_photometry.scatter(dataset_dates,dataset_ztfg ,label='ztf-g',marker='v', alpha=0.75, c='green', edgecolor='black',s=200)
        ax_photometry.scatter(dataset_dates, dataset_ztfr, label='ztf-r',marker='^', alpha=0.75,c='red', edgecolor='black',s=200)
        ax_photometry.scatter(dataset_dates, dataset_ztfi, label='ztf-i',marker='>', alpha=0.75, c='yellow', edgecolor='black',s=200)
        ax_photometry.set_title('Photometry')
        ax_photometry.grid(alpha=0.15)
        ax_photometry.legend()
        ## spectra
        ax_spectra = fig.add_subplot(gs[0, 4])
        ax_spectra.plot(spectra[0], label='Spectra', color='yellowgreen')
        ax_spectra.set_title('Spectra')
        ax_spectra.grid(alpha=0.15)
        ax_spectra.legend()
    
        plt.tight_layout()
        plt.show()
        
        
    ## for this, you need to get object id from more_info.give_me_the_original_ish_alert(train_dataset, train_index, data_df, TRAIN_DATA_PATH)
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
        
    
    