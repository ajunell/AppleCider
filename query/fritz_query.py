import requests
from typing import Mapping, Optional
import urllib
import urllib.parse
import os
import numpy as np
import importlib
import multiprocessing
import collections
from tqdm.auto import tqdm
import time
import pandas as pd 


token = 'data_redacted'

def api(
    method: str,
    endpoint: str,
    data: Optional[Mapping] = None,
    base_url: str = "https://fritz.science",
    ):
    
    ''' see skyportal.io/docs/api.html#description/path-parameters for more info! '''

    headers = {"Authorization": f"token {token}"}
    response = requests.request(
      method.upper(),
      urllib.parse.urljoin(base_url, endpoint),
      json=data,
      headers=headers
    )
    return response


# data length check
def query_classification(obj_id):
    
    """
    Given a list of objects, get Fritz classification that meets requirements: classif probability >= 0.6, ml classifed = False 
    
    this function does this in the most deluded way possible: multiple if, else statements to check progressively older classifs
    if the most recent does not meet the reqs. sorry in advance, if there are more than 4 classifications that don't meet the reqs
    those objects will be saved to a specific list denotating that... you will have to query their classifications differently <3
    
    """

    min_prob = 0.59 # min probability of classification
    prob_nan = np.nan ; prob_none = None ; prob_blank = " " # account for no prob. on classification 
    con_ml = True ; tax_id = 8 # extra conditions on classifications (we don't want these)
    
    obj_pass = [] # object classifications that meet [insert] requirements
    #fail_resp = [] # objects that get no response from api: failed status code or rate limit exceeded
    fail_req = [] # object classification that DON'T meet [insert] requirements
    fail_data = [] # no classification in fritz / no data 
    _2big2fail = [] # objects w/more than 4 classifications that don't meet requirements
    #fail_other = [] # misc failures (in the very weird case they happen)

    
    for object_id in tqdm(obj_id, desc='Processing Objects', leave=True):
        time.sleep(0.05)
        response = api("get", f"api/sources/{object_id}/classifications")
        
        # deal w/invalid / error  
        if response.status_code == 400:
            tqdm.write(f'Response {object_id} failed')
            tqdm.write(f'JSON response: {response.json()}')
            fail_data.append(object_id) # save object w/invalid response from api
            time.sleep(0.25) # to avoid rate limit
            continue
        # deal w/rate limit 
        if response.status_code == 429: # see: https://skyportal.io/docs/api.html
            tqdm.write(f"Request rate limit exceeded at [ {object_id} ] ; sleeping 1s before trying again...")
            fail_data.append(object_id) # save object w/invalid response from api
            time.sleep(1) # to avoid rate limit
            continue

        data = response.json().get("data", None)
        if response.json().get("data", None) is None: # (account for "Expecting value: line 1 column 1 (char 0)" error)
            tqdm.write(f"Request data [ {object_id} ] FAILED")
            fail_data.append(object_id)
            continue
        
        data_length = len(data) # use this to find most recent classification
        time.sleep(0.01)        # to avoid rate limit
        
        # depends on number of entries / classifications
        if data_length == 0: # NO classification in Fritz
            fail_data.append(object_id)
            
        elif data_length == 1: # ONE classification in Fritz
            data_1 = data[(data_length) - 1 ]
            classif = data_1['classification'] #classif_prob = data['probability']
            prob_min = data_1['probability']
            prob_missing = (data_1['probability'] is prob_nan or data_1['probability'] is prob_none or data_1['probability'] is prob_blank)
            condition_tm = (data_1['taxonomy_id'] is not tax_id and data_1['ml'] is not con_ml)
            # check if classification meets pre-reqs
            if condition_tm and (prob_missing or prob_min > min_prob):
                tqdm.write(f"{object_id} classified with  prob:{data_1['probability']} as {classif}")
                obj_pass.append((object_id, data_1['classification']))
            
            else: # object classification failed and since there's only one classification... to the fail_req pile
                tqdm.write(f"{object_id} FAILED requirements. taxonomy: {data_1['taxonomy_id']}, prob: {data_1['probability']}, class: {classif} ,ml: {data_1['ml']}")
                fail_req.append((object_id,classif))
        
        elif data_length >= 2: # MORE THAN ONE classification in Fritz
            data_2 = data[(data_length) - 1 ]
            classif = data_2['classification']
            prob_min = data_2['probability']
            prob_missing = data_2['probability'] is prob_nan or data_2['probability'] is prob_none or data_2['probability'] is prob_blank
            condition_tm = (data_2['taxonomy_id'] is not tax_id and data_2['ml'] is not con_ml)
            
            if condition_tm and (prob_missing or prob_min > min_prob):
                tqdm.write(f"{object_id} classified with prob:{data_2['probability']} as {classif}")
                obj_pass.append((object_id, data_2['classification']))
                
            else: # MORE THAN TWO classifications in Fritz
                data_old = data[(data_length)-2] 
                classif = data_old['classification']
                prob_min = data_old['probability']
                prob_missing = data_old['probability'] is prob_nan or data_old['probability'] is prob_none or data_old['probability'] is prob_blank
                condition_tm = (data_old['taxonomy_id'] is not tax_id and data_old['ml'] is not con_ml)

                if condition_tm and (prob_missing or prob_min > min_prob):
                    tqdm.write(f"{object_id} classified with prob:{data_old['probability']} as {classif}")
                    obj_pass.append((object_id, data_old['classification']))
                
                elif ((data_length) - 3 ) < 0: # no more data, will fail next loop
                    tqdm.write(f"{object_id} FAILED requirements (d3). taxonomy: {data_old['taxonomy_id']}, prob: {data_old['probability']}, class: {classif} ,ml: {data_old['ml']}")
                    fail_req.append((object_id,classif))
                    
                else: # MORE THAN THREE classifications in Fritz
                    data_older = data[(data_length) - 3 ]
                    classif = data_older['classification']
                    prob_min = data_older['probability']
                    prob_missing = (data_older['probability'] is prob_nan or data_older['probability'] is prob_none or data_older['probability'] is prob_blank)
                    condition_tm = (data_older['taxonomy_id'] is not tax_id and data_older['ml'] is not con_ml)
                    
                    if condition_tm and (prob_missing or prob_min > min_prob):
                        tqdm.write(f"{object_id} classified with prob:{data_older['probability']} as {classif}")
                        obj_pass.append((object_id, data_older['classification']))
                        
                    elif ((data_length) - 4 ) < 0: # no more data, would fail next loop
                        tqdm.write(f"{object_id} FAILED requirements (d4). taxonomy: {data_older['taxonomy_id']}, prob: {data_older['probability']}, class: {classif} ,ml: {data_older['ml']}")
                        fail_req.append((object_id,classif))
                        
                    else: # MORE THAN FOUR classifications in Fritz
                        tqdm.write(f"{object_id} TOO BIG TO FAIL! TOO MUCH DATA (d4). taxonomy: {data_older['taxonomy_id']}, prob: {data_older['probability']}, class: {classif} ,ml: {data_older['ml']}")
                        _2big2fail.append((object_id, data_length))
        
        else:
            fail_data.append(object_id)
            tqdm.write(f"{object_id} FAILED OR SOMETHING")
            

    return obj_pass, fail_req, fail_data, _2big2fail


def query_spectra(obj_id):
    '''
    Parameters
    ----------
    obj_id : list
        list of ZTF IDs you want to search for
    Returns 
    ----------
    data_spectra: list
        ZTF ID, ZTF ID, wavelength, flux, observed date, observed MJD,
        instrument name, telescope name,
        data length (aka number spectrums that are in Fritz)
        !!! warning: this only saves the MOST RECENT spectra in Fritz !!!
    skipped_list: list
        ZTF IDs where api response failed
    rate_limit_list: list
        ZTF IDs that request rate limit was exceeded at
    no_spectra: list
        ZTF IDs that don't have spectra
    '''

    data_spectra = [] # most recent spectra results for each object
    skipped_list = [] # api response failed for object
    rate_limit_list = [] # object is victim of rate limit
    no_spectra = []
    
    for object_id in tqdm(obj_id, desc='Processing Objects',leave=True):
        response = api("get", f"api/sources/{object_id}/spectra")
        time.sleep(0.20) # any higher than 0.20 and rate limit 
        if response.status_code == 400:
            tqdm.write(f'Response {object_id} failed')
            tqdm.write(f'JSON response: {response.json()}')
            skipped_list.append(object_id)
            time.sleep(0.05)
            continue
        if response.status_code == 429:
            tqdm.write(f"Request rate limit exceeded at [ {object_id} ] ; sleeping 2s before trying again...")
            time.sleep(2)
            skipped_listt.append(object_id)
            continue
        data = response.json().get("data", None)
        
        if data is None: # (account for "Expecting value: line 1 column 1 (char 0)" error)
            tqdm.write(f"Request data [ {object_id} ] FAILED")
            skipped_list.append(object_id)
            time.sleep(0.025)
            continue

        data_spec = data['spectra']
        data_length = len(data_spec)

        if data_length == 0:
            no_spectra.append(object_id)
            tqdm.write(f"{object_id} NO spectra")
        elif data_length >= 1:
            tqdm.write(f"{object_id} {data_length} spectra")
            
            # save the object ID, wavelengths, fluxes, observed date (month, year, date & in MJD), 
            #          instrument name, telescope name, # of spectra taken
            for i in range(len(data_spec[0]['fluxes'])):
                data_spectra.append((object_id, data_spec[0]['wavelengths'][i],data_spec[0]['fluxes'][i],
                                 data_spec[0]['observed_at'], data_spec[0]['observed_at_mjd'] ,data_spec[0]['instrument_name'],
                                 data_spec[0]['telescope_name'], data_length ))
        else:
            print(f"{object_id} no data?")
            no_spectra.append(object_id)
            tqdm.write(f"Request data [ {object_id} ] FAILED")

    return data_spectra, skipped_list, no_spectra


def save_obj_spectra_to_obj_folders(all_spectra_df, obj_id_list, base_path):
    '''
    Parameters
    ----------
    all_spectra_df : pd.DataFrame
        MUST BE DF CONTAINING ALL SPECTRA INFO FOR ALL OBJECTS IN THE DATASET IN ONE DATAFRAME
        (i.e. USE 
                    
                data_spectra, _, _, _,_  =  query_spectra() 
                all_spectra_df =  pd.DataFrame(data_spectra, columns=['obj_id','wavelength',
                                                        'flux' ,'observed date', 'observed MJD', 'telescope name',
                                                        'data length])
                                                    )
    
    obj_id_list : list of unique ZTF IDs in all_spectra_df
    base_path : directory to save .csv to, inside of this will be folders for each object ID

    Returns 
    ----------
    you get individual .csv for each object's spectra <3
    '''

    for obj_id in tqdm(obj_id_list, desc='Processing Objects',leave=True):
        objDirectory = os.path.join(base_path, obj_id)
        obj_str = str(obj_id)
        filename = 'spectra.csv'
        
        obj_spectra_df = all_spectra_df[all_spectra_df['ZTFID'] == obj_id]
        obj_spectra_df.to_csv(os.path.join(objDirectory, 'spectra.csv'), index=False)
        tqdm.write(f"saved {obj_id}")
 


def query_info(obj_id):
    '''
    Parameters
    ----------
    ZTF ID : list
    
    Returns 
    ----------
    list with ZTF ID, RA, Dec, RA error, Dec Error,TNS ID, 
    host galaxies, gal lat, gal lon, 
    '''
    data_info = [] 
    no_info = [] # api response failed for object
    #rate_limit_list = [] ; skipped_list = []
    skipped_list = []
    
    for object_id in tqdm(obj_id, desc='Processing Objects',leave=True):

        response = api("get", f"api/sources/{object_id}")
        time.sleep(0.11)

        if response.status_code == 400:
            tqdm.write(f'Response {object_id} failed')
            tqdm.write(f'JSON response: {response.json()}')
            skipped_list.append(object_id)
            time.sleep(0.15)
            continue
        if response.status_code == 429: # see: https://skyportal.io/docs/api.html
            tqdm.write(f"Request rate limit exceeded at [ {object_id} ] ; sleeping 2s before trying again...")
            time.sleep(2)
            skipped_list.append(object_id)
            continue
        data = response.json().get("data", None)
        
        if data is None: # (account for "Expecting value: line 1 column 1 (char 0)" error)
            tqdm.write(f"Request data [ {object_id} ] FAILED")
            skipped_list.append(object_id)
            time.sleep(0.025)
            continue

        if len(data) == 0:
            no_info.append(object_id)
            tqdm.write(f"{object_id} NO info")
        else: 
            data_ra = data['ra'] ; data_dec = data['dec']
            data_ra_err = data['ra_err'] ; data_dec_err = data['dec_err']

            data_tns_name = data['tns_name']
            data_host_gal = data['galaxies']
            data_gal_lat = data['gal_lat'] ; data_gal_lon = data['gal_lon']
            data_lum_distance = data['luminosity_distance']

            data_info.append((object_id, data_tns_name, data_ra, data_dec, data_ra_err, data_dec_err,
                              data_host_gal, data_gal_lat, data_gal_lon, data_lum_distance))

    return data_info, no_info, skipped_list