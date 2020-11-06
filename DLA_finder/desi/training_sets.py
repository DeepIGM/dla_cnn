""" Code to build/load/write DESI Training sets"""

'''
1. Load up the Sightlines
2. Split into samples of kernel length
3. Grab DLAs and non-DLA samples
4. Hold in memory or write to disk??
5. Convert to TF Dataset
'''

import multiprocessing
from multiprocessing import Pool
import itertools

import numpy as np

from dla_cnn.Timer import Timer
from dla_cnn.desi.DesiMock import DesiMock
from dla_cnn.spectra_utils import get_lam_data
from dla_cnn.training_set import select_samples_50p_pos_neg
from dla_cnn.desi.defs import REST_RANGE,kernel

def split_sightline_into_samples(sightline, REST_RANGE=REST_RANGE, kernel=kernel):
    """
    Split the sightline into a series of snippets, each with length kernel

    Parameters
    ----------
    sightline: dla_cnn.data_model.Sightline
    REST_RANGE: list
    kernel: int, optional

    Returns
    -------

    """
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    kernelrangepx = int(kernel/2) #200
    #samplerangepx = int(kernel*pos_sample_kernel_percent/2) 
    #consider boundaries
    cut=((np.nonzero(ix_dla_range)[0])>=kernelrangepx)&((np.nonzero(ix_dla_range)[0])<=(len(lam)-kernelrangepx-1))
     
    #ix_dlas = [(np.abs(lam[ix_dla_range]-dla.central_wavelength).argmin()) for dla in sightline.dlas]
    #coldensity_dlas = [dla.col_density for dla in sightline.dlas]       # column densities matching ix_dlas

    # FLUXES - Produce a 400 matrix of flux values
    fluxes_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0][cut])))
    lam_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(lam), np.nonzero(ix_dla_range)[0][cut])))
    # Return
    return fluxes_matrix, sightline.classification[cut], sightline.offsets[cut], sightline.column_density[cut],lam_matrix

# Returns indexes of pos & neg samples that are 50% positive and 50% negative and no boarder
def select_samples_50p_pos_neg(sightline,kernel=kernel):
    """
    For a given sightline, generate the indices for DLAs and for without
    Split 50/50 to have equal representation
    Parameters
    ----------
    classification: np.ndarray
        Array of classification values.  1=DLA; 0=Not; -1=not analyzed
    Returns
    -------
    idx: np.ndarray
        positive + negative indices
    """
    #classification = data[1]
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso)
    kernelrangepx = int(kernel/2)
    cut=((np.nonzero(ix_dla_range)[0])>=kernelrangepx)&((np.nonzero(ix_dla_range)[0])<=(len(lam)-kernelrangepx-1))
    newclassification=sightline.classification[cut]
    num_pos = np.sum(newclassification==1, dtype=np.float64)
    num_neg = np.sum(newclassification==0, dtype=np.float64)
    n_samples = int(min(num_pos, num_neg))

    r = np.random.permutation(len(newclassification))

    pos_ixs = r[newclassification[r]==1][0:n_samples]
    neg_ixs = r[newclassification[r]==0][0:n_samples]
    # num_total = data[0].shape[0]
    # ratio_neg = num_pos / num_neg

    # pos_mask = classification == 1      # Take all positive samples

    # neg_ixs_by_ratio = np.linspace(1,num_total-1,round(ratio_neg*num_total), dtype=np.int32) # get all samples by ratio
    # neg_mask = np.zeros((num_total),dtype=np.bool) # create a 0 vector of negative samples
    # neg_mask[neg_ixs_by_ratio] = True # set the vector to positives, selecting for the appropriate ratio across the whole sightline
    # neg_mask[pos_mask] = False # remove previously positive samples from the set
    # neg_mask[classification == -1] = False # remove border samples from the set, what remains is still in the right ratio

    # return pos_mask | neg_mask
    return np.hstack((pos_ixs,neg_ixs))
    
def prepare_training_test_set(ids_train, ids_test,
                                      train_save_file="../data/localize_train.npy",
                                      test_save_file="../data/localize_test.npy",
                                      ignore_sightline_markers={},
                                      save=False):
    """
    Build a Training set for DESI

    and a test set, as desired

    Args:
        ids_train: list
        ids_test: list (can be empty)
        train_save_file: str
        test_save_file: str
        ignore_sightline_markers:
        save: bool

    Returns:

    """
    num_cores = multiprocessing.cpu_count() - 1
    p = Pool(num_cores, maxtasksperchild=10)  # a thread pool we'll reuse

    # Training data
    with Timer(disp="read_sightlines"):
        sightlines_train=[]
        for ii in ids_train:
            sightlines_train.append(specs.get_sightline(ids_train[ii],'all',True,True))
        # add the ignore markers to the sightline
        for s in sightlines_train:
            if hasattr(s.id, 'sightlineid') and s.id.sightlineid >= 0:
                s.data_markers = ignore_sightline_markers[s.id.sightlineid] if ignore_sightline_markers.has_key(
                    s.id.sightlineid) else []
    with Timer(disp="split_sightlines_into_samples"):
        data_split = p.map(split_sightline_into_samples, sightlines_train)
    with Timer(disp="select_samples_50p_pos_neg"):
        sample_masks = p.map(select_samples_50p_pos_neg, data_split[1])
    with Timer(disp="zip and stack"):
        zip_data_masks = zip(data_split, sample_masks)
        data_train = {}
        data_train['flux'] = np.vstack([d[0][m] for d, m in zip_data_masks])
        data_train['labels_classifier'] = np.hstack([d[1][m] for d, m in zip_data_masks])
        data_train['labels_offset'] = np.hstack([d[2][m] for d, m in zip_data_masks])
        data_train['col_density'] = np.hstack([d[3][m] for d, m in zip_data_masks])
    if save:
        with Timer(disp="save train data files"):
            save_tf_dataset(train_save_file, data_train)

    # Same for test data if it exists
    data_test = {}
    if len(ids_test) > 0:
        sightlines_test = p.map(read_sightline, ids_test)
        data_split = map(split_sightline_into_samples, sightlines_test)
        sample_masks = map(select_samples_50p_pos_neg, data_split)
        zip_data_masks = zip(data_split, sample_masks)
        data_test['flux'] = np.vstack([d[0][m] for d, m in zip_data_masks])
        data_test['labels_classifier'] = np.hstack([d[1][m] for d, m in zip_data_masks])
        data_test['labels_offset'] = np.hstack([d[2][m] for d, m in zip_data_masks])
        data_test['col_density'] = np.hstack([d[3][m] for d, m in zip_data_masks])
        if save:
            save_tf_dataset(test_save_file, data_test)

    # Return
    return data_train, data_test
