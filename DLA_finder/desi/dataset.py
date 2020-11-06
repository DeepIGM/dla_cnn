import numpy as np
import scipy.signal as signal
from dla_cnn.desi.training_sets import split_sightline_into_samples 
from dla_cnn.desi.preprocess import label_sightline
from dla_cnn.spectra_utils import get_lam_data
from dla_cnn.training_set import select_samples_50p_pos_neg

def make_datasets(sightlines,validate=True):
    """
    Generate training set or validation set for DESI.
    
    Parameters:
    -----------------------------------------------
    sightlines: list of 'dla_cnn.data_model.Sightline' object, the sightlines should be preprocessed.
    validate: bool
    
    Returns
    -----------------------------------------------
    dataset:dict, the training set contains flux and 3 labels, the validation set contains flux, lam, 3 labels and DLAs' data.
    
    """
    dataset={}
    for sightline in sightlines:
        wavelength_dlas=[dla.central_wavelength for dla in sightline.dlas]
        coldensity_dlas=[dla.col_density for dla in sightline.dlas]   
        label_sightline(sightline)
        data_split=split_sightline_into_samples(sightline)
        if validate:
            flux=np.vstack([data_split[0]])
            labels_classifier=np.hstack([data_split[1]])
            labels_offset=np.hstack([data_split[2]])
            col_density=np.hstack([data_split[3]])
            lam=np.vstack([data_split[4]])
            dataset[sightline.id]={'FLUX':flux,'lam':lam,'labels_classifier':  labels_classifier, 'labels_offset':labels_offset , 'col_density': col_density,'wavelength_dlas':wavelength_dlas,'coldensity_dlas':coldensity_dlas} 
        else:
            sample_masks=select_samples_50p_pos_neg(sightline)
            if sample_masks !=[]:
                flux=np.vstack([data_split[0][m] for m in sample_masks])
                labels_classifier=np.hstack([data_split[1][m] for m in sample_masks])
                labels_offset=np.hstack([data_split[2][m] for m in sample_masks])
                col_density=np.hstack([data_split[3][m] for m in sample_masks])
            dataset[sightline.id]={'FLUX':flux,'labels_classifier':labels_classifier,'labels_offset':labels_offset,'col_density': col_density}
    return dataset
        
def smooth_flux(flux):
    """
    Smooth flux using median filter.
    
    Parameters:
    -----------------------------------------------
    flux: np.ndarry flux for every kernel
    
    Return:
    -----------------------------------------------
    flux_matrix:list, 4-dimension flux data
    
    """
    flux_matrix=[]
    for sample in flux:#sample是片段flux
                smooth3=signal.medfilt(sample,3)
                smooth7=signal.medfilt(sample,7)
                smooth15=signal.medfilt(sample,15)
                flux_matrix.append(np.array([sample,smooth3,smooth7,smooth15]))
    return flux_matrix
    
def make_smoothdatasets(sightlines,validate=True):
    """
    Generate smoothed training set or validation set for DESI.
    
    Parameters:
    -----------------------------------------------
    sightlines: list of 'dla_cnn.data_model.Sightline' object, the sightlines should be preprocessed.
    validate: bool
    
    Returns
    -----------------------------------------------
    dataset:dict, the training set contains smoothed flux and 3 labels, the validation set contains smoothed flux, lam, 3 labels and DLAs' data.
    
    """
    dataset={}
    for sightline in sightlines:
        wavelength_dlas=[dla.central_wavelength for dla in sightline.dlas]
        coldensity_dlas=[dla.col_density for dla in sightline.dlas]   
        label_sightline(sightline)
        data_split=split_sightline_into_samples(sightline)
        if validate:
            flux=np.vstack([data_split[0]])
            labels_classifier=np.hstack([data_split[1]])
            labels_offset=np.hstack([data_split[2]])
            col_density=np.hstack([data_split[3]])
            lam=np.vstack([data_split[4]])
            flux_matrix=smooth_flux(flux)
            dataset[sightline.id]={'FLUXMATRIX':flux_matrix,'lam':lam,'labels_classifier':  labels_classifier, 'labels_offset':labels_offset , 'col_density': col_density,'wavelength_dlas':wavelength_dlas,'coldensity_dlas':coldensity_dlas} 
        else:
            sample_masks=select_samples_50p_pos_neg(sightline)
            if sample_masks !=[]:
                flux=np.vstack([data_split[0][m] for m in sample_masks])
                labels_classifier=np.hstack([data_split[1][m] for m in sample_masks])
                labels_offset=np.hstack([data_split[2][m] for m in sample_masks])
                col_density=np.hstack([data_split[3][m] for m in sample_masks])
                flux_matrix=smooth_flux(flux)
                dataset[sightline.id]={'FLUXMATRIX':flux_matrix,'labels_classifier':labels_classifier,'labels_offset':labels_offset,'col_density': col_density}
    return dataset
