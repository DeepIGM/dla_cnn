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
import random

from dla_cnn.Timer import Timer
from dla_cnn.desi.io import read_sightline
from dla_cnn.spectra_utils import get_lam_data
from dla_cnn.training_set import select_samples_50p_pos_neg
#from dla_cnn.desi.load_fits_files import load_DesiMocks,sightline_retriever

def split_sightline_into_samples(sightline, REST_RANGE, kernel=400):
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
    #samplerangepx = int(kernel*pos_sample_kernel_percent/2) #60
    kernelrangepx = int(kernel/2) #200
    cut=((np.nonzero(ix_dla_range)[0])>=kernelrangepx)&((np.nonzero(ix_dla_range)[0])<=(len(lam)-kernelrangepx-1))
    #ix_dlas = [(np.abs(lam[ix_dla_range]-dla.central_wavelength).argmin()) for dla in sightline.dlas]
    #coldensity_dlas = [dla.col_density for dla in sightline.dlas]       # column densities matching ix_dlas

    # FLUXES - Produce a 1748x400 matrix of flux values
    fluxes_matrix = np.vstack(map(lambda f,r:f[r-kernelrangepx:r+kernelrangepx],
                                  zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0][cut])))
    # Return
    return fluxes_matrix, sightline.classification[cut], sightline.offsets[cut], sightline.column_density[cut]


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
        sightlines_train = p.map(read_sightline, ids_train)
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

def split_bins(data_array,num):
    """
    Split bins for snr_array,zdla_array,nhi_array
    -------------------------------------------------------------------------------------
    parameters；
    data_array:np.ndarray
    num:int
    --------------------------------------------------------------------------------------
    return:
    bins_index : list
    """
    d=np.ptp(data_array)/num
    datamin=np.amin(data_array)
    bins_index=[]
    for ii in range(0,num):
        #bins_data.append(data_array[(data_array>=d*ii+datamin)&(data_array<=d*(ii+1)+datamin)])
        bins_index.append((data_array>=d*ii+datamin)&(data_array<=d*(ii+1)+datamin))
    
    return bins_index
    
def snr_uniform(snr,idarray,binnum,num,snrrange):
    """
    Uniform snr in snrrange
    Parameter
    ---------
    snr:np.ndarray
    idarray:np.ndarray
    binnum: int
    num:int
    snrrange:list
             lowest snr,highest snr
    
    Return
    ---------
    samplesnr_id：np.ndarray
    """
    new_snr=snr[(snr<=snrrange[1])&(snr>=snrrange[0])]
    new_id=idarray[(snr<=snrrange[1])&(snr>=snrrange[0])]
    snr_index=split_bins(new_snr,binnum)
    samplesnr_id=[]
    for jj in range(0,binnum):
        samplesnr_id.append(random.sample(list(new_id[snr_index[jj]]),num))
    samplesnr_id=np.array(samplesnr_id).ravel()
    return samplesnr_id

def uniform_z_nhi(idarray,z_bins,nhi_bins,num,Mocks):
    """
    Uniform zdla,nhi,get sightline id
    -------------------------------------------------------------------------------
    parameters；
    idlist:numpy.ndarray
           sightlineid array after uniform snr
    z_bins:int
    nhi_bins:int
    num:int
    Mocks:list
          all files in DesiMock form 
    -------------------------------------------------------------------------------
    return:
    nhisample:list
    zsample:list
    sampleid:sightline idlist
    sample_dla_id:np.ndarray
    """
    
    nhi=[]
    zdla=[]
    dlaid=[]
    mockid=[]
    for ii in idlist:
        sightline=sightline_retriever(ii,Mocks)
        for jj in sightline.dlas:
            if 912<=(jj.central_wavelength/(1+sightline.z_qso))<=1220:#dla restframe[912,1220]
                nhi.append(jj.col_density) 
                zdla.append(float(jj.central_wavelength/1215.6701-1))
                mockid.append(ii)
                dlaid.append(str(ii)+jj.id)
    #cut NHI(20.3-21.5),zdla with rsd,dlaid
    nhicut=(np.array(nhi)>=20.3)&(np.array(nhi)<=21.2)&(np.array(zdla)>=2.13)#dla central wavelength>3800
    nhi_dla=np.array(nhi)[nhicut]
    zdla=np.array(zdla)[nhicut]
    mockid=np.array(mockid)[nhicut]
    dlaid=np.array(dlaid)[nhicut]
    #z split bins, get index
    z_index=split_bins(zdla,z_bins)
    #nhi split bins,get index
    nhi_index=[]
    for ii in range(0,z_bins):
        nhi_index.append(split_bins(nhi_dla[z_index[ii]],nhi_bins))
    #get bins id
    dlaid_bins=[]
    for i in range(0,z_bins):
        for j in range(0,nhi_bins):
            dlaid_bins.append(dlaid[z_index[i]][nhi_index[i][j]])
    #uniform idsample in every bins
    sample_dla_id=[]
    for jj in range(0,len(dlaid_bins)):
        sample_dla_id.append(np.random.choice(list(dlaid_bins[jj]),num))
    sample_dla_id=np.array(sample_dla_id).ravel()
    nhisample=[]
    zsample=[]
    sampleid=[]
    for i in range(0,len(sample_dla_id)): 
        index=np.where(dlaid==sample_dla_id[i])
        nhisample.append(nhi_dla[index])
        zsample.append(zdla[index])
        sampleid.append(mockid[index])
    #return
    return nhisample,zsample,sampleid,sample_dla_id

def select_samples_50p_pos_neg(sightline,kernel):
    """
    For a given sightline, generate the indices for DLAs and for without
    Split 50/50 to have equal representation
    Parameters
    ----------
    sightline:dla_cnn.data_model.Sightline.Sightline object
    kernel:int
    Returns
    -------
    idx: np.ndarray
        positive + negative indices
    """
  
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE=[900,1346])
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

