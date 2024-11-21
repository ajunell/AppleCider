import numpy as np
import pandas as pd
import george
import os
import scipy.optimize as op
from functools import partial
from astropy.table import Table, vstack
import pickle


pb_wavelengths = {
    'ztfg': 4800.,
    'ztfr': 6400.,
    'ztfi': 7900.,
}

filters = ['ztfg', 'ztfr', 'ztfi']
gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)
inverse_gp_wavelengths = {v: k for k, v in pb_wavelengths.items()}

# Some of these functions are from "Paying attention to astronomical transients: introducing the time-series transformer for photometric classification"
# ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ http://github.com/tallamjr/astronet ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆
def fit_2d_gp(obj_data, kernel=None, return_kernel=False, global_fit=False, **kwargs):
    '''Fit a 2D Gaussian process
    
    taken from:  
    ☆ ☆ ☆ ☆ ☆ http://github.com/tallamjr/astronet/blob/master/astronet/preprocess.py ☆ ☆ ☆ ☆ ☆
    
    Parameters
    ----------
    obj_data : pd.DataFrame
        Time, flux and flux error of the data (specific filter of an object).
    return_kernel : bool, default = False
        Whether to return the used kernel.
    pb_wavelengths: dict
        Mapping of the passband wavelengths for each filter used.
    kwargs : dict
        Additional keyword arguments that are ignored at the moment. We allow
        additional keyword arguments so that the various functions that
        call this one can be called with the same arguments.
    
    Returns
    -------
    kernel: george.gp.GP.kernel, optional
        The kernel used to fit the GP.
    gp_predict : functools.partial of george.gp.GP
        The GP instance that was used to fit the object.
    
    '''
    
    if kernel is None:
        guess_length_scale = 20.0
        signal_to_noises = np.abs(obj_data.flux) / np.sqrt(
            obj_data.flux_error ** 2 + (1e-2 * np.max(obj_data.flux)) ** 2
        )
        scale = np.abs(obj_data.flux[signal_to_noises.idxmax()])
        kernel = (0.5 * scale) ** 2 * george.kernels.Matern32Kernel([
            guess_length_scale ** 2, 6000 ** 2], ndim=2)
        kernel.freeze_parameter("k2:metric:log_M_1_1")

    obj_times = obj_data.mjd.astype(float)
    obj_flux = obj_data.flux.astype(float)
    obj_flux_error = obj_data.flux_error.astype(float)
    obj_wavelengths = obj_data['filter'].map(pb_wavelengths)

    def neg_log_like(p):  # Objective function: negative log-likelihood
        gp.set_parameter_vector(p)
        loglike = gp.log_likelihood(obj_flux, quiet=True)
        return -loglike if np.isfinite(loglike) else 1e25

    def grad_neg_log_like(p):  # Gradient of the objective function.
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(obj_flux, quiet=True)

    gp = george.GP(kernel)
    default_gp_param = gp.get_parameter_vector()
    x_data = np.vstack([obj_times, obj_wavelengths]).T
    gp.compute(x_data, obj_flux_error)

    bounds = [(0, np.log(1000 ** 2))]
    bounds = [(default_gp_param[0] - 10, default_gp_param[0] + 10)] + bounds
    results = op.minimize(neg_log_like, gp.get_parameter_vector(),
                          jac=grad_neg_log_like, method="L-BFGS-B",
                          bounds=bounds, tol=1e-6)

    if results.success:
        gp.set_parameter_vector(results.x)
    else:
        # Fit failed. Print out a warning, and use the initial guesses for fit
        # parameters.
        obj = obj_data['obj_id'][0]
        print("GP fit failed for {}! Using guessed GP parameters.".format(obj))
        gp.set_parameter_vector(default_gp_param)

    gp_predict = partial(gp.predict, obj_flux)

    if return_kernel:
        return kernel, gp_predict
    else:
        return gp_predict


def predict_2d_gp(gp_predict, gp_times, gp_wavelengths):
    """Outputs the predictions of a Gaussian Process.

    adapted from:  
    ☆ ☆ ☆ ☆ ☆ http://github.com/tallamjr/astronet/blob/master/astronet/preprocess.py ☆ ☆ ☆ ☆ ☆

    Parameters
    ----------
    gp_predict : functools.partial of george.gp.GP
        The GP instance that was used to fit the object.
    gp_times : numpy.ndarray
        Times to evaluate the Gaussian Process at.
    gp_wavelengths : numpy.ndarray
        Wavelengths to evaluate the Gaussian Process at.

    Returns
    -------
    obj_gps : pandas.core.frame.DataFrame, optional
        Time, flux and flux error of the fitted Gaussian Process.
    ...
    """

    unique_wavelengths = np.unique(gp_wavelengths)
    number_gp = len(gp_times)
    obj_gps = []
    for wavelength in unique_wavelengths:
        gp_wavelengths = np.ones(number_gp) * wavelength
        pred_x_data = np.vstack([gp_times, gp_wavelengths]).T
        pb_pred, pb_pred_var = gp_predict(pred_x_data, return_var=True)
        # stack the GP results in a array momentarily
        obj_gp_pb_array = np.column_stack((gp_times, pb_pred, np.sqrt(pb_pred_var)))
        obj_gp_pb = Table(
            [
                obj_gp_pb_array[:, 0],
                obj_gp_pb_array[:, 1],
                obj_gp_pb_array[:, 2],
                [wavelength] * number_gp,
            ],
            names=["mjd", "flux", "flux_error", "filter"],
        )
        if len(obj_gps) == 0:
            obj_gps = obj_gp_pb
        else: 
            obj_gps = vstack((obj_gps, obj_gp_pb))
            
    obj_gps = obj_gps.to_pandas()
    return obj_gps


def process_gaussian(df, kernel=None, number_gp=200, save=False, name=''):
    
    res_df = pd.DataFrame()
    time_window = 90  # Define a fixed time window of 90 MJD

    for obj_id in df['obj_id'].unique():
        obj_df = df[df['obj_id'] == obj_id]
        type_obj = obj_df['type'].values[0]
        obj_df.reset_index(drop=True, inplace=True)

        available_filters = obj_df['filter'].unique()
        gp_wavelengths = np.vectorize(pb_wavelengths.get)(available_filters)
        inverse_gp_wavelengths = {v: k for k, v in pb_wavelengths.items() if k in available_filters}

        # Ensure the time window is from 0 to 90 MJD
        gp_times = np.linspace(0, time_window, number_gp)
        # Fit the Gaussian Process
        gp_predict = fit_2d_gp(obj_df, kernel=kernel)
        # Get predictions
        obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
        # Map back the filter names
        obj_gps['filter'] = obj_gps['filter'].map(inverse_gp_wavelengths)
        # Add zero-padding for flux and flux_error after the last actual data point
        last_mjd = obj_df['mjd'].max()
        obj_gps.loc[obj_gps['mjd'] > last_mjd, ['flux', 'flux_error']] = 0
        # Pivot the table for better organization
        obj_gps = obj_gps.pivot_table(index=['mjd'], columns='filter', values=['flux', 'flux_error'])
        obj_gps = obj_gps.reset_index()
        obj_gps.columns = [col[0] if col[0] == 'mjd' else '_'.join(col).strip() for col in obj_gps.columns.values]
        obj_gps['type'] = type_obj
        obj_gps['obj_id'] = obj_id

        res_df = pd.concat([res_df, obj_gps])

    res_df.reset_index(drop=True, inplace=True)

    if save:
        types_str = '_'.join(df['type'].unique()) if hasattr(df['type'].unique(), '__iter__') else str(df['type'].unique())
        filename = f'_gp_{types_str}.csv'
        filename = filename.replace(' ', '_')
        filename = filename.replace('/', '_') # added, without it SN Ib doesn't save

        from pathlib import Path
        directory = '_gp_type/'
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        file_path = os.path.join(directory, filename)
        res_df.to_csv(file_path, index=False)

        print(f'File {filename} saved successfully')
    return res_df

def save_kernel(kernel, filename):
    with open(filename, 'wb') as f:
        pickle.dump(kernel, f)
    print(f'Kernel saved to {filename}')

def load_kernel(filename):
    with open(filename, 'rb') as f:
        kernel = pickle.load(f)
    print(f'Kernel loaded from {filename}')
    return kernel
