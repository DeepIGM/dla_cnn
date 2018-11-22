""" Module for loading spectra, either fake or real
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import os, urllib, math, json, timeit, multiprocessing, gc, sys, warnings, re, pickle, gzip, h5py, itertools, glob, time
from traceback import print_exc
import pdb

from pkg_resources import resource_filename

from scipy.stats import chisquare
from scipy.optimize import minimize

from astropy.io import fits
from astropy.table import Table
from multiprocessing import Process, Value, Array, Pool
from dla_cnn.data_model.Sightline import Sightline
from dla_cnn.data_model.Dla import Dla
from dla_cnn.data_model.Id_GENSAMPLES import Id_GENSAMPLES
from dla_cnn.data_model.Id_DR12 import Id_DR12
from dla_cnn.data_model.Id_old_DR12 import Id_old_DR12  # FITS files
from dla_cnn.data_model.Id_DR7 import Id_DR7
from dla_cnn.data_model.Prediction import Prediction
from dla_cnn.data_model.DataMarker import Marker
import code, traceback, threading
from dla_cnn.localize_model import predictions_ann as predictions_ann_c2
import scipy.signal as signal
from scipy.spatial.distance import cdist
from operator import itemgetter, attrgetter, methodcaller
from dla_cnn.Timer import Timer
from mpl_toolkits.axes_grid.inset_locator import inset_axes

# Raise warnings to errors for debugging
import warnings
#warnings.filterwarnings('error')


# DLAs from the DR9 catalog range from 920 to 1214, adding 120 on the right for variance in ly-a
# the last number is the number of pixels in SDSS sightlines that span the range
# REST_RANGE = [920, 1334, 1614]
# REST_RANGE = [911, 1346, 1696]
REST_RANGE = [900, 1346, 1748]
cache = {}              # Cache for files and resources that should be opened once and kept open
TF_DEVICE = os.getenv('TF_DEVICE', '')
lock = threading.Lock()

default_model = resource_filename('dla_cnn', "models/model_gensample_v7.1")


# Rads fits file locally based on plate-mjd-fiber or online if option is set
def read_fits_file(plate, mjd, fiber, fits_base_dir="../../BOSS_dat_all", download_if_notfound=False):
    # Open the fits file
    fits_filename = "%s/spec-%04d-%05d-%04d.fits" % (fits_base_dir, int(plate), int(mjd), int(fiber))
    if os.path.isfile(fits_filename) and fits_base_dir is not None:
        return read_fits_filename(fits_filename)
    elif download_if_notfound:
        url = "http://dr12.sdss3.org/sas/dr12/boss/spectro/redux/v5_7_0/spectra/%04d/spec-%04d-%05d-%04d.fits" % \
              (plate, plate, mjd, fiber)
        r = urllib.urlretrieve(url)
        data = read_fits_filename(r[0])
        os.remove(r[0])
        return data
    else:
        raise Exception("File not found in [%s], and download_if_notfound is False" % fits_base_dir)


def read_fits_filename(fits_filename):
    with fits.open(fits_filename) as fits_file:
        data1 = fits_file[1].data.copy()
        z_qso = fits_file[3].data['LINEZ'][0].copy()

        raw_data = {}
        # Pad loglam and flux_normalized to sufficiently below 920A rest that we don't have issues falling off the left
        (loglam_padded, flux_padded) = pad_loglam_flux(data1['loglam'], data1['flux'], z_qso)
        raw_data['flux'] = flux_padded
        raw_data['loglam'] = loglam_padded
        raw_data['plate'] = fits_file[2].data['PLATE'].copy()
        raw_data['mjd'] = fits_file[2].data['MJD'].copy()
        raw_data['fiber'] = fits_file[2].data['FIBERID'].copy()
        raw_data['ra'] = fits_file[2].data['RA'].copy()
        raw_data['dec'] = fits_file[2].data['DEC'].copy()

        for hdu in fits_file:
            del hdu.data
    gc.collect()    # Workaround for issue with mmap numpy not releasing the fits file: https://goo.gl/IEfAPh

    return raw_data, z_qso


def read_custom_hdf5(sightline):
    """ Read custom HDF5 files made for this project
    Parameters
    ----------
    sightline : Sightline

    Returns
    -------

    """
    global cache
    fs = sightline.id.hdf5_datafile
    json_datafile = sightline.id.json_datafile
    if ~cache.has_key(fs) or ~cache.has_key(json_datafile):
        with lock:
            if not cache.has_key(fs) or not cache.has_key(json_datafile):
                # print "Cache miss: [%s] and/or [%s] not found in cache" % (fs, json_datafile)
                cache[fs] = h5py.File(fs, "r")
                cache[json_datafile] = json.load(open(json_datafile))
    f = cache[fs]
    j = cache[json_datafile]

    ix = sightline.id.ix
    lam, flux, sig, _ = f['data'][ix]

    # print "DEBUG> read_custom_hdf5 [%s] --- index: [%d]" % (sightline.id.hdf5_datafile, ix)

    # Trim leading or training 0's and non finite values to clean up the data
    # Can't use np.non_zero here because of the Inf values
    first = 0
    for i in flux:
        if i == 0 or ~np.isfinite(i):
            first += 1
        else:
            break
    last = len(lam)
    for i in flux[::-1]:
        if i == 0 or ~np.isfinite(i):
            last -= 1
        else:
            break
    lam = lam[first:last]
    flux = flux[first:last]
    sig = sig[first:last]
    assert np.all(np.isfinite(lam) & np.isfinite(flux) & np.isfinite(sig))

    loglam = np.log10(lam)

    # z_qso
    if len(f['meta'].shape) == 1:                   # This was for the dr5 no-dla sightlines lacking a JSON file
        z_qso = f['cut_meta']['zem_GROUP'][ix]
    else:
        meta = json.loads(f['meta'].value)
        # Two different runs named this key different things
        z_qso = meta['headers'][sightline.id.ix]['zem'] \
            if meta['headers'][sightline.id.ix].has_key('zem') else meta['headers'][sightline.id.ix]['zem_GROUP']

    # Pad loglam and flux_normalized to sufficiently below 920A rest that we don't have issues falling off the left
    (loglam_padded, flux_padded, sig_padded) = pad_loglam_flux(loglam, flux, z_qso, sig=sig)
    assert(np.all(np.logical_and(np.isfinite(loglam_padded), np.isfinite(flux_padded))))

    # sightline id
    sightline.id.sightlineid = j[str(ix)]['sl'] if j[str(ix)].has_key('sl') else -1

    sightline.dlas = []
    for dla_ix in range(0,int(j[str(ix)]['nDLA'])):
        central_wavelength = (1 + float(j[str(ix)][str(dla_ix)]['zabs'])) * 1215.67
        col_density = float(j[str(ix)][str(dla_ix)]['NHI'])
        sightline.dlas.append(Dla(central_wavelength, col_density))
    sightline.flux = flux_padded
    sightline.sig = sig_padded
    sightline.loglam = loglam_padded
    sightline.z_qso = z_qso

    if not validate_sightline(sightline):
        print("error validating sightline! bug! exiting")
        exit()

    return sightline


# Reads spectra out of IgmSpec library for DR7 or DR12 (plate & fiber only)
def read_igmspec(plate, fiber, ra=-1, dec=-1, mjd=-1, table_name='SDSS_DR7'):
    with open(os.devnull, 'w') as devnull:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Hack to avoid specdb spamming us with print statements
            stdout = sys.stdout
            sys.stdout = devnull

            from specdb.specdb import IgmSpec  # Custom package only used in this optional read function

            # global igmtables, igmsp
            global cache
            if table_name not in cache.keys():  # ~cache.has_key(table_name):
                with lock:
                    #if ~cache.has_key(table_name):
                    if table_name not in cache:
                        cache['igmsp'] = IgmSpec()
                        cache[table_name] = Table(cache['igmsp'].hdf[table_name + "/meta"].value)
            igmsp = cache['igmsp']
            mtbl = cache[table_name]

            print("Plate/Fiber: ", plate, fiber)
            plate = int(plate)
            fiber = int(fiber)

            # Find plate/fiber
            if table_name == 'SDSS_DR7':
                imt = np.where((mtbl['PLATE'] == plate) & (mtbl['FIBER'] == fiber))[0]
            elif table_name == 'BOSS_DR12':
                imt = np.where((mtbl['PLATE'] == plate) & (mtbl['FIBERID'] == fiber) & (mtbl['MJD'] == mjd))[0]
            igmid = mtbl['IGM_ID'][imt]
            # print "imt, igmid", imt, igmid, type(imt), type(igmid), type(mtbl), np.shape(mtbl), "end-print"
            assert np.shape(igmid)[0] == 1, "Expected igmid to contain exactly 1 value, found %d" % np.shape(igmid)[0]

            raw_data = {}
            # spec, meta = igmsp.idb.grab_spec([table_name], igmid)
            # spec, meta = igmsp.allspec_of_ID(igmid, groups=[table_name])
            spec, meta = igmsp.spectra_from_ID(igmid, groups=[table_name])

            z_qso = meta['zem_GROUP'][0]
            flux = np.array(spec[0].flux)
            sig = np.array(spec[0].sig)
            loglam = np.log10(np.array(spec[0].wavelength))
            (loglam_padded, flux_padded, sig_padded) = pad_loglam_flux(loglam, flux, z_qso, sig=sig)
            # Sanity check that we're getting the log10 values
            assert np.all(loglam < 10), "Loglam values > 10, example: %f" % loglam[0]

            raw_data['flux'] = flux_padded
            raw_data['sig'] = sig_padded
            raw_data['loglam'] = loglam_padded
            raw_data['plate'] = plate
            raw_data['mjd'] = 0
            raw_data['fiber'] = fiber
            raw_data['ra'] = ra
            raw_data['dec'] = dec
            assert np.shape(raw_data['flux']) == np.shape(raw_data['loglam'])
            sys.stdout = stdout

    return raw_data, z_qso


def pad_loglam_flux(loglam, flux, z_qso, kernel=1800, sig=None):
    # kernel = 1800    # Overriding left padding to increase it
    assert np.shape(loglam) == np.shape(flux)
    pad_loglam_upper = loglam[0] - 0.0001
    pad_loglam_lower = (math.floor(math.log10(REST_RANGE[0] * (1 + z_qso)) * 10000) - kernel / 2) / 10000
    pad_loglam = np.linspace(pad_loglam_lower, pad_loglam_upper, max(0, int((pad_loglam_upper - pad_loglam_lower + 0.0001) * 10000)), dtype=np.float32)
    pad_value = np.mean(flux[0:50])
    flux_padded = np.hstack((pad_loglam*0+pad_value, flux))
    loglam_padded = np.hstack((pad_loglam, loglam))
    assert (10**loglam_padded[0])/(1+z_qso) <= REST_RANGE[0]
    # Error array
    if sig is not None:
        sig_padded = np.hstack((pad_loglam*0+pad_value, sig))
        return loglam_padded, flux_padded, sig_padded
    else:
        return loglam_padded, flux_padded


def scan_flux_sample(flux_normalized, loglam, z_qso, central_wavelength, #col_density, plate, mjd, fiber, ra, dec,
                     exclude_positive_samples=False, kernel=400, stride=5,
                     pos_sample_kernel_percent=0.3, testing=None):
    # Split from rest frame 920A to 1214A (the range of DLAs in DR9 catalog)
    # pos_sample_kernel_percent is the percent of the kernel where positive samples can be found
    # e.g. the central wavelength is within this percentage of the center of the kernel window

    # Pre allocate space for generating samples
    samples_buffer = np.zeros((10000, kernel), dtype=np.float32)
    offsets_buffer = np.zeros((10000,), dtype=np.float32)
    buffer_count = 0

    lam, lam_rest, ix_dla_range = get_lam_data(loglam, z_qso, REST_RANGE)
    ix_from = np.nonzero(ix_dla_range)[0][0]
    ix_to = np.shape(lam_rest)[0] - np.nonzero(np.flipud(ix_dla_range))[0][0]
    ix_central = np.nonzero(lam >= central_wavelength)[0][0]

    assert (ix_to > ix_central)

    # Scan across the data set generating negative samples
    # (skip positive samples where lam is near the central wavelength)
    for position in range(ix_from, ix_to, stride):
        if abs(position - ix_central) > kernel * pos_sample_kernel_percent:
            # Add a negative sample (not within pos_sample_kernel_percent of the central_wavelength)
            try:
                samples_buffer[buffer_count, :] = flux_normalized[position - kernel // 2:position - kernel // 2 + kernel]
            except (IndexError, ValueError):  # Running off the red side of the spectrum (I think)
                # Kludge to pad with data at end of spectrum
                samples_buffer[buffer_count, :] = flux_normalized[-kernel:]
            offsets_buffer[buffer_count] = 0
            buffer_count += 1
        elif not exclude_positive_samples:
            # Add a positive sample (is within pos_sample_kernel_percent of the central_wavelength)
            samples_buffer[buffer_count, :] = flux_normalized[position - kernel // 2:position - kernel // 2 + kernel]
            offsets_buffer[buffer_count] = position - ix_central
            buffer_count += 1

    # return samples_buffer[0:buffer_count, :]
    return samples_buffer[0:buffer_count, :], offsets_buffer[0:buffer_count] #neg_flux, neg_offsets


def read_sightline(id):
    sightline = Sightline(id=id)
    if isinstance(id, Id_old_DR12, ):
        data1, z_qso = read_fits_file(id.plate, id.mjd, id.fiber)
        sightline.id.ra = data1['ra']
        sightline.id.dec = data1['dec']
        sightline.flux = data1['flux']
        sightline.loglam = data1['loglam']
        sightline.z_qso = z_qso
    elif isinstance(id, (Id_DR7, Id_DR12)):
        if isinstance(id, Id_DR7):
            data1, z_qso = read_igmspec(id.plate, id.fiber, id.ra, id.dec)
        elif isinstance(id, Id_DR12):
            data1, z_qso = read_igmspec(id.plate, id.fiber, id.ra, id.dec, mjd=id.mjd, table_name='BOSS_DR12')
        sightline.id.ra = data1['ra']
        sightline.id.dec = data1['dec']
        sightline.flux = data1['flux']
        sightline.sig = data1['sig']
        sightline.loglam = data1['loglam']
        sightline.z_qso = z_qso
    elif isinstance(id, Id_GENSAMPLES):
        read_custom_hdf5(sightline)
    else:
        raise Exception("%s not implemented yet" % type(id).__name__)
    return sightline


def preprocess_data_from_dr9(kernel=400, stride=3, pos_sample_kernel_percent=0.3,
                             train_keys_csv="../data/dr9_train_set.csv",
                             test_keys_csv="../data/dr9_test_set.csv"):
    dr9_train = np.genfromtxt(train_keys_csv, delimiter=',')
    dr9_test = np.genfromtxt(test_keys_csv, delimiter=',')

    # Dedup ---(there aren't any in dr9_train, so skipping for now)
    # dr9_train_keys = np.vstack({tuple(row) for row in dr9_train[:,0:3]})

    sightlines_train = [Sightline(Id_DR12(s[0],s[1],s[2]),[Dla(s[3],s[4])]) for s in dr9_train]
    sightlines_test  = [Sightline(Id_DR12(s[0],s[1],s[2]),[Dla(s[3],s[4])]) for s in dr9_test]

    prepare_localization_training_set(kernel, stride, pos_sample_kernel_percent,
                                      sightlines_train, sightlines_test)


# Length 1 for non array elements
def pseudolen(p):
    return len(p) if hasattr(p,'__len__') else 1


# Set save_file parameters to null to return the results and not write them to disk
def preprocess_gensample_from_single_hdf5_file(kernel=400, stride=3, pos_sample_kernel_percent=0.3, percent_test=0.0,
                                               datafile='../data/training_100',
                                               save_file="../data/localize",
                                               ignore_sightline_markers=None):#"../data/ignore_data_dr5_markers.csv"):
    hdf5_datafile = datafile + ".hdf5"
    json_datafile = datafile + ".json"
    train_save_file = save_file + "_train" if percent_test > 0.0 else save_file
    test_save_file = save_file + "_test"

    with open(json_datafile, 'r') as fj:
        n = len(json.load(fj))
        n_train = int((1-percent_test)*n)
        ids_train = [Id_GENSAMPLES(i, hdf5_datafile, json_datafile) for i in range(0,n_train)]
        ids_test  = [Id_GENSAMPLES(i, hdf5_datafile, json_datafile) for i in range(n_train,n)]

    markers_csv = np.loadtxt(ignore_sightline_markers, delimiter=',') if ignore_sightline_markers != None else []
    markers = {}
    for m in markers_csv:
        markers[m[0]] = [] if not markers.has_key(m[0]) else markers[m[0]]
        markers[m[0]].append(Marker(m[1]))


    prepare_localization_training_set(kernel, stride, pos_sample_kernel_percent,
                                      ids_train, ids_test,
                                      train_save_file=train_save_file,
                                      test_save_file=test_save_file,
                                      ignore_sightline_markers=markers)


def preprocess_overlapping_dla_sightlines_from_gensample(kernel=400, stride=3, pos_sample_kernel_percent=0.3, percent_test=0.0,
                                               datafile='../data/gensample_hdf5_files/dlas/training*',
                                               save_file = "../data/gensample/train_overlapdlas"):
    hdf5_datafiles = sorted(glob.glob(datafile + ".hdf5"))
    json_datafiles = sorted(glob.glob(datafile + ".json"))

    ids = []
    for hdf5_datafile, json_datafile in zip(hdf5_datafiles, json_datafiles):
        with open(json_datafile, 'r') as f:
            j = json.load(f)
            for i in range(5000):
                n_dlas = j[str(i)]['nDLA']
                if n_dlas > 1:
                    dlas = np.array([j[str(i)][str(n)]['zabs'] for n in range(n_dlas)]) # array of DLA zabs values
                    dlas = np.reshape(dlas, (len(dlas),1))          # reshape for use in cdist
                    distances = cdist(dlas, dlas, 'cityblock')      # get distances between each dla

                    if np.min(distances[distances>0.0][:]) < 0.2:                  # if there's at least one pair
                        ids.append(Id_GENSAMPLES(ix=i, hdf5_datafile=hdf5_datafile, json_datafile=json_datafile))

    prepare_localization_training_set(kernel, stride, pos_sample_kernel_percent,
                                      ids, [],
                                      train_save_file=save_file,
                                      test_save_file=None)


def prepare_localization_training_set(kernel, stride, pos_sample_kernel_percent,
                                      ids_train, ids_test,
                                      train_save_file="../data/localize_train.npy",
                                      test_save_file="../data/localize_test.npy",
                                      ignore_sightline_markers={}):
    num_cores = multiprocessing.cpu_count() - 1
    p = Pool(num_cores, maxtasksperchild=10)       # a thread pool we'll reuse

    # Training data
    with Timer(disp="read_sightlines"):
        sightlines_train = p.map(read_sightline, ids_train)
        # add the ignore markers to the sightline
        for s in sightlines_train:
            if hasattr(s.id, 'sightlineid') and s.id.sightlineid >= 0:
                s.data_markers = ignore_sightline_markers[s.id.sightlineid] if ignore_sightline_markers.has_key(s.id.sightlineid) else []
    with Timer(disp="split_sightlines_into_samples"):
        data_split = p.map(split_sightline_into_samples, sightlines_train)
    with Timer(disp="select_samples_50p_pos_neg"):
        sample_masks = p.map(select_samples_50p_pos_neg, data_split)
    with Timer(disp="zip and stack"):
        zip_data_masks = zip(data_split, sample_masks)
        data_train = {}
        data_train['flux'] = np.vstack([d[0][m] for d,m in zip_data_masks])
        data_train['labels_classifier'] = np.hstack([d[1][m] for d,m in zip_data_masks])
        data_train['labels_offset'] = np.hstack([d[2][m] for d,m in zip_data_masks])
        data_train['col_density'] = np.hstack([d[3][m] for d,m in zip_data_masks])
    with Timer(disp="save train data files"):
        save_dataset(train_save_file, data_train)

    # Same for test data if it exists
    if len(ids_test) > 0:
        sightlines_test = p.map(read_sightline, ids_test)
        data_split = map(split_sightline_into_samples, sightlines_test)
        sample_masks = map(select_samples_50p_pos_neg, data_split)
        zip_data_masks = zip(data_split, sample_masks)
        data_test = {}
        data_test['flux'] = np.vstack([d[0][m] for d,m in zip_data_masks])
        data_test['labels_classifier'] = np.hstack([d[1][m] for d,m in zip_data_masks])
        data_test['labels_offset'] = np.hstack([d[2][m] for d,m in zip_data_masks])
        data_test['col_density'] = np.hstack([d[3][m] for d,m in zip_data_masks])
        save_dataset(test_save_file, data_test)


# Receives data in the tuple form returned from split_sightline_into_samples:
# (fluxes_matrix, classification, offsets_array, column_density)
# Returns indexes of pos & neg samples that are 50% positive and 50% negative and no boarder
def select_samples_50p_pos_neg(data):
    classification = data[1]
    num_pos = np.sum(classification==1, dtype=np.float64)
    num_neg = np.sum(classification==0, dtype=np.float64)
    n_samples = int(min(num_pos, num_neg))

    r = np.random.permutation(len(classification))

    pos_ixs = r[classification[r]==1][0:n_samples]
    neg_ixs = r[classification[r]==0][0:n_samples]
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


def validate_sightline(sightline):
    # check that all DLAs are in range
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    ix_from = np.nonzero(ix_dla_range)[0][0]
    ix_to = np.shape(lam_rest)[0] - np.nonzero(np.flipud(ix_dla_range))[0][0]
    for dla in sightline.dlas:
        ix_central = np.nonzero(lam >= dla.central_wavelength)[0][0]

        if ix_to > ix_central and ix_from < ix_central:
            continue
        else:
            return False
    return True


def save_dataset(save_file, data):
    print("Writing %s.npy to disk" % save_file)
    # np.save(save_file+".npy", data['flux'])
    # data['flux'] = None
    # print "Writing %s.pickle to disk" % save_file
    # with gzip.GzipFile(filename=save_file+".pickle", mode='wb', compresslevel=2) as f:
    #     pickle.dump([data], f, protocol=-1)
    np.savez_compressed(save_file,
                        flux=data['flux'],
                        labels_classifier=data['labels_classifier'],
                        labels_offset=data['labels_offset'],
                        col_density=data['col_density'])


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def split_sightline_into_samples(sightline,
                                 kernel=400, pos_sample_kernel_percent=0.3):
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    samplerangepx = int(kernel*pos_sample_kernel_percent/2) #60
    kernelrangepx = int(kernel/2) #200
    ix_dlas = [(np.abs(lam[ix_dla_range]-dla.central_wavelength).argmin()) for dla in sightline.dlas]
    coldensity_dlas = [dla.col_density for dla in sightline.dlas]       # column densities matching ix_dlas

    # FLUXES - Produce a 1748x400 matrix of flux values
    fluxes_matrix = np.vstack(map(lambda f,r:f[r-kernelrangepx:r+kernelrangepx],
                                  zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0])))

    # CLASSIFICATION (1 = positive sample, 0 = negative sample, -1 = border sample not used
    # Start with all samples negative
    classification = np.zeros((REST_RANGE[2]), dtype=np.float32)
    # overlay samples that are too close to a known DLA, write these for all DLAs before overlaying positive sample 1's
    for ix_dla in ix_dlas:
        classification[ix_dla-samplerangepx*2:ix_dla+samplerangepx*2+1] = -1
        # Mark out Ly-B areas
        lyb_ix = sightline.get_lyb_index(ix_dla)
        classification[lyb_ix-samplerangepx:lyb_ix+samplerangepx+1] = -1
    # mark out bad samples from custom defined markers
    for marker in sightline.data_markers:
        assert marker.marker_type == Marker.IGNORE_FEATURE              # we assume there are no other marker types for now
        ixloc = np.abs(lam_rest - marker.lam_rest_location).argmin()
        classification[ixloc-samplerangepx:ixloc+samplerangepx+1] = -1
    # overlay samples that are positive
    for ix_dla in ix_dlas:
        classification[ix_dla-samplerangepx:ix_dla+samplerangepx+1] = 1

    # OFFSETS & COLUMN DENSITY
    offsets_array = np.full([REST_RANGE[2]], np.nan, dtype=np.float32)     # Start all NaN markers
    column_density = np.full([REST_RANGE[2]], np.nan, dtype=np.float32)
    # Add DLAs, this loop will work from the DLA outward updating the offset values and not update it
    # if it would overwrite something set by another nearby DLA
    for i in range(int(samplerangepx+1)):
        for ix_dla,j in zip(ix_dlas,range(len(ix_dlas))):
            offsets_array[ix_dla+i] = -i if np.isnan(offsets_array[ix_dla+i]) else offsets_array[ix_dla+i]
            offsets_array[ix_dla-i] =  i if np.isnan(offsets_array[ix_dla-i]) else offsets_array[ix_dla-i]
            column_density[ix_dla+i] = coldensity_dlas[j] if np.isnan(column_density[ix_dla+i]) else column_density[ix_dla+i]
            column_density[ix_dla-i] = coldensity_dlas[j] if np.isnan(column_density[ix_dla-i]) else column_density[ix_dla-i]
    offsets_array = np.nan_to_num(offsets_array)
    column_density = np.nan_to_num(column_density)

    # fluxes is 1748x400 of fluxes
    # classification is 1 / 0 / -1 for DLA/nonDLA/border
    # offsets_array is offset
    return fluxes_matrix, classification, offsets_array, column_density


def get_lam_data(loglam, z_qso, REST_RANGE):
    lam = 10.0 ** loglam
    lam_rest = lam / (1.0 + z_qso)
    ix_dla_range = np.logical_and(lam_rest >= REST_RANGE[0], lam_rest <= REST_RANGE[1])

    # ix_dla_range may be 1 pixels shorter or longer due to rounding error, we force it to a consistent size here
    size_ix_dla_range = np.sum(ix_dla_range)
    assert size_ix_dla_range >= REST_RANGE[2] - 2 and size_ix_dla_range <= REST_RANGE[2] + 2, \
        "Size of DLA range assertion error, size_ix_dla_range: [%d]" % size_ix_dla_range
    b = np.nonzero(ix_dla_range)[0][0]
    if size_ix_dla_range < REST_RANGE[2]:
        # Add a one to the left or right sides, making sure we don't exceed bounds on the left
        ix_dla_range[max(b - 1, 0):max(b - 1, 0) + REST_RANGE[2]] = 1
    if size_ix_dla_range > REST_RANGE[2]:
        ix_dla_range[b + REST_RANGE[2]:] = 0  # Delete 1 or 2 zeros from right side
    assert np.sum(ix_dla_range) == REST_RANGE[2], \
        "Size of ix_dla_range: %d, %d, %d, %d, %d" % \
        (np.sum(ix_dla_range), b, REST_RANGE[2], size_ix_dla_range, np.nonzero(np.flipud(ix_dla_range))[0][0])

    return lam, lam_rest, ix_dla_range


# Expects a sightline with the prediction object complete, updates the peaks_ixs of the sightline object
def compute_peaks(sightline):
    PEAK_THRESH = 0.2                   # Threshold to accept a peak
    PEAK_SEPARATION_THRESH = 0.1        # Peaks must be separated by a valley at least this low

    # Translate relative offsets to histogram
    offset_to_ix = np.arange(len(sightline.prediction.offsets)) + sightline.prediction.offsets
    offset_to_ix[offset_to_ix < 0] = 0
    offset_to_ix[offset_to_ix >= len(sightline.prediction.offsets)] = len(sightline.prediction.offsets)
    offset_hist, ignore_offset_range = np.histogram(offset_to_ix, bins=np.arange(0,len(sightline.prediction.offsets)+1))

    # Somewhat arbitrary normalization
    offset_hist = offset_hist / 80.0

    po = np.pad(offset_hist, 2, 'constant', constant_values=np.mean(offset_hist))
    offset_conv_sum = (po[:-4] + po[1:-3] + po[2:-2] + po[3:-1] + po[4:])
    smooth_conv_sum = signal.medfilt(offset_conv_sum, 9)
    # ensures a 0 value at the beginning and end exists to avoid an unnecessarily pathalogical case below
    smooth_conv_sum[0] = 0
    smooth_conv_sum[-1] = 0

    peaks_ixs = []
    while True:
        peak = np.argmax(smooth_conv_sum)   # Returns the first occurace of the max
        # exit if we're no longer finding valid peaks
        if smooth_conv_sum[peak] < PEAK_THRESH:
            break
        # skip this peak if it's off the end or beginning of the sightline
        if peak <= 10 or peak >= REST_RANGE[2]-10:
            smooth_conv_sum[max(0,peak-15):peak+15] = 0
            continue
        # move to the middle of the peak if there are multiple equal values
        ridge = 1
        while smooth_conv_sum[peak] == smooth_conv_sum[peak+ridge]:
            ridge += 1
        peak = peak + ridge//2
        peaks_ixs.append(peak)

        # clear points around the peak, that is, anything above PEAK_THRESH in order for a new DLA to be identified the peak has to dip below PEAK_THRESH
        clear_left = smooth_conv_sum[0:peak+1] < PEAK_SEPARATION_THRESH # something like: 1 0 0 1 1 1 0 0 0 0
        clear_left = np.nonzero(clear_left)[0][-1]+1                    # Take the last value and increment 1
        clear_right = smooth_conv_sum[peak:] < PEAK_SEPARATION_THRESH   # something like 0 0 0 0 1 1 1 0 0 1
        clear_right = np.nonzero(clear_right)[0][0]+peak                # Take the first value & add the peak offset
        smooth_conv_sum[clear_left:clear_right] = 0

    sightline.prediction.peaks_ixs = peaks_ixs
    sightline.prediction.offset_hist = offset_hist
    sightline.prediction.offset_conv_sum = offset_conv_sum
    return sightline


def process_catalog_dr7(csv_plate_mjd_fiber="../data/dr7_test_set.csv",
                        kernel_size=400, pfiber=None, make_pdf=False,
                        model_checkpoint=default_model,
                        output_dir="../tmp/visuals_dr7"):
    """ Generates a SDSS DR7 DLA catalog from plate/mjd/fiber from a CSV file
    Parameters
    ----------
    csv_plate_mjd_fiber
    kernel_size
    pfiber
    make_pdf
    model_checkpoint
    output_dir

    Returns
    -------

    """
    csv = Table.read(csv_plate_mjd_fiber)
    ids = [Id_DR7(c[0],c[1],c[2],c[3]) for c in csv]
    if pfiber is not None:
        plates = np.array([iid.plate for iid in ids])
        fibers = np.array([iid.fiber for iid in ids])
        imt = np.where((plates==pfiber[0]) & (fibers==pfiber[1]))[0]
        if len(imt) != 1:
            print("Plate/Fiber not in DR7!!")
            pdb.set_trace()
        else:
            ids = [ids[imt[0]]]
    process_catalog(ids, kernel_size, model_checkpoint, make_pdf=make_pdf,
                    CHUNK_SIZE=500, output_dir=output_dir)

# Generates a catalog from plate/mjd/fiber from a CSV file
def process_catalog_dr12(csv_plate_mjd_fiber="../data/dr12_test_set.csv",
                        kernel_size=400, pfiber=None, make_pdf=False,
                        model_checkpoint=default_model,
                        output_dir="../tmp/visuals_dr12"):
    #csv = np.genfromtxt(csv_plate_mjd_fiber, delimiter=',')
    csv = Table.read(csv_plate_mjd_fiber)
    ids = [Id_DR12(c[0],c[1],c[2],c[3],c[4]) for c in csv]
    if pfiber is not None:
        plates = np.array([iid.plate for iid in ids])
        fibers = np.array([iid.fiber for iid in ids])
        imt = np.where((plates==pfiber[0]) & (fibers==pfiber[1]))[0]
        if len(imt) != 1:
            print("Plate/Fiber not in DR12!!")
            pdb.set_trace()
        else:
            ids = [ids[imt[0]]]
    process_catalog(ids, kernel_size, model_checkpoint, CHUNK_SIZE=500,
                    make_pdf=make_pdf, output_dir=output_dir)


def process_catalog_gensample(gensample_files_glob="../data/gensample_hdf5_files/test_mix_23559_10000.hdf5",
                              json_files_glob=     "../data/gensample_hdf5_files/test_mix_23559_10000.json",
                              kernel_size=400, debug=False,
                              model_checkpoint=default_model,
                              output_dir="../tmp/visuals_gensample96451/"):
    """ Generate a DLA catalog from a general sample
    Usually used for validation

    Parameters
    ----------
    gensample_files_glob
    json_files_glob
    kernel_size
    model_checkpoint
    output_dir

    Returns
    -------

    """
    expanded_datafiles = sorted(glob.glob(gensample_files_glob))
    expanded_json = sorted(glob.glob(json_files_glob))
    ids = []
    for hdf5_datafile, json_datafile in zip(expanded_datafiles, expanded_json):
        with open(json_datafile, 'r') as fj:
            n = len(json.load(fj))
            ids.extend([Id_GENSAMPLES(i, hdf5_datafile, json_datafile) for i in range(0, n)])
        process_catalog(ids, kernel_size, model_checkpoint, output_dir=output_dir, debug=debug)


# Process a directory of fits files in format ".*plate-mjd-fiber.*"
def process_catalog_fits_pmf(fits_dir="../../BOSS_dat_all",
                             model_checkpoint=default_model,
                             output_dir="../tmp/visuals/",
                             kernel_size=400):
    ids = []
    for f in glob.glob(fits_dir + "/*.fits"):
        match = re.match(r'.*-(\d+)-(\d+)-(\d+)\..*', f)
        if not match:
            print("Match failed on: ", f)
            exit()
        ids.append(Id_DR12(int(match.group(1)),int(match.group(2)),int(match.group(3))))

    process_catalog(ids, kernel_size=kernel_size, model_path=model_checkpoint, output_dir=output_dir)


def process_catalog_csv_pmf(csv="../data/boss_catalog.csv",
                            model_checkpoint=default_model,
                            output_dir="../tmp/visuals/",
                            kernel_size=400):
    pmf = np.loadtxt(csv, dtype=np.int64, delimiter=',')
    ids = [Id_DR12(row[0],row[1],row[2]) for row in pmf]
    process_catalog(ids, model_path=model_checkpoint, output_dir=output_dir, kernel_size=kernel_size)

# This function processes a full catalog of sightlines, it's not meant to call directly,
# for each catalog there will be a helper function dedicated to that catalog type:
#   process_catalog_gensample
#   process_catalog_dr12
#   process_catalog_dr5

def process_catalog(ids, kernel_size, model_path="", debug=False,
                    CHUNK_SIZE=1000, output_dir="../tmp/visuals/",
                    make_pdf=False, num_cores=None):
    from dla_cnn.plots import generate_pdf
    from dla_cnn.absorption import add_abs_to_sightline
    if num_cores is None:
        num_cores = multiprocessing.cpu_count() - 1
    # num_cores = 24
    # p = None
    p = Pool(num_cores)  # a thread pool we'll reuse
    if debug:
        num_cores = 1
        p = None
    sightlines_processed_count = 0

    sightline_results = []  # array of map objects containing the classification, and an array of DLAs

    ids.sort(key=methodcaller('id_string'))

    # We'll handle the full process in batches so as to not exceed memory constraints
    done = False
    for sss,ids_batch in enumerate(np.array_split(ids, np.arange(CHUNK_SIZE,len(ids),CHUNK_SIZE))):
        num_sightlines = len(ids_batch)
        #if sss < 46:  # debugging
        #    sightlines_processed_count += num_sightlines
        #    continue
        if done:
            break
        # # Workaround for segfaults occuring in matplotlib, kill multiprocess pool every iteration
        # if p is not None:
        #     p.close()
        #     p.join()
        #     time.sleep(5)

        report_timer = timeit.default_timer()

        # Batch read files
        process_timer = timeit.default_timer()
        print("Reading {:d} sightlines with {:d} cores".format(num_sightlines, num_cores))
        if debug:
            sightlines_batch = []
            for iid in ids_batch:
                sightlines_batch.append(read_sightline(iid))
        else:
            sightlines_batch = p.map(read_sightline, ids_batch)
        print("Spectrum/Fits read done in {:0.1f}".format(timeit.default_timer() - process_timer))

        ##################################################################
        # Process model
        ##################################################################
        print("Model predictions begin")
        fluxes = np.vstack([scan_flux_sample(s.flux, s.loglam, s.z_qso, -1, stride=1)[0] for s in sightlines_batch])
        #fluxes = np.vstack([scan_flux_sample(s.flux, s.loglam, s.z_qso, -1, stride=1, testing=s)[0] for s in sightlines_batch])
        with open(model_path+"_hyperparams.json",'r') as f:
            hyperparameters = json.load(f)
        loc_pred, loc_conf, offsets, density_data_flat = predictions_ann_c2(hyperparameters, fluxes, model_path)

        # Add results from predictions and peaks_data to data model for easier processing later.
        for sl, lp, lc, of, dd in zip(sightlines_batch,
                                      np.split(loc_pred, num_sightlines),
                                      np.split(loc_conf, num_sightlines),
                                      np.split(offsets, num_sightlines),
                                      np.split(density_data_flat, num_sightlines)):
            sl.prediction = Prediction(loc_pred=lp, loc_conf=lc, offsets=of, density_data=dd)

        with Timer(disp="Compute peaks"):
            sightlines_batch = map(compute_peaks, sightlines_batch)
            sightlines_batch = sorted(sightlines_batch, key=lambda s: s.id.id_string())

        ##################################################################
        # Process output for each sightline
        ##################################################################
        assert num_sightlines * REST_RANGE[2] == density_data_flat.shape[0]
        for sightline in sightlines_batch:
            smoothed_sample = sightline.prediction.smoothed_loc_conf()

            # Add absorbers
            add_abs_to_sightline(sightline)

            # Store classification level data in results
            sightline_json = ({
                'id':           sightline.id.id_string(),
                'ra':           float(sightline.id.ra),
                'dec':          float(sightline.id.dec),
                'z_qso':        float(sightline.z_qso),
                'num_dlas':     len(sightline.dlas),
                'num_subdlas':  len(sightline.subdlas),
                'num_lyb':      len(sightline.lybs),
                'dlas':         sightline.dlas,
                'subdlas':      sightline.subdlas,
                'lyb':          sightline.lybs
            })

            sightline_results.append(sightline_json)

        ##################################################################
        # Process pdfs for each sightline
        ##################################################################
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # print "Processing PDFs"
        if make_pdf:
             #p.map(generate_pdf, zip(sightlines_batch, itertools.repeat(output_dir)))  # TODO
             p.starmap(generate_pdf, zip(sightlines_batch, itertools.repeat(output_dir)))  # TODO

        print("Processed {:d} sightlines for reporting on {:d} cores in {:0.2f}s".format(
              num_sightlines, num_cores, timeit.default_timer() - report_timer))

        runtime = timeit.default_timer() - process_timer
        print("Processed {:d} of {:d} in {:0.0f}s - {:0.2f}s per sample".format(
              sightlines_processed_count + num_sightlines, len(ids), runtime, runtime/num_sightlines))
        sightlines_processed_count += num_sightlines
        if debug:
            done = True


    # Write JSON string
    with open(output_dir + "/predictions.json", 'w') as outfile:
        json.dump(sightline_results, outfile, indent=4)

# Add S/N after the fact
def add_s2n_after(ids, json_file, debug=False, CHUNK_SIZE=1000):
    from linetools import utils as ltu
    from dla_cnn.absorption import get_s2n_for_absorbers   # Needs to be here

    # Load json file
    predictions = ltu.loadjson(json_file)
    jids = [ii['id'] for ii in predictions]

    num_cores = multiprocessing.cpu_count() - 2
    p = Pool(num_cores)  # a thread pool we'll reuse
    sightlines_processed_count = 0

    # IDs
    ids.sort(key=methodcaller('id_string'))
    for sss,ids_batch in enumerate(np.array_split(ids, np.arange(CHUNK_SIZE,len(ids),CHUNK_SIZE))):
        num_sightlines = len(ids_batch)
        # Read batch
        process_timer = timeit.default_timer()
        print("Reading {:d} sightlines with {:d} cores".format(num_sightlines, num_cores))
        sightlines_batch = p.map(read_sightline, ids_batch)
        print("Done reading")

        for sightline in sightlines_batch:
            jidx = jids.index(sightline.id.id_string())
            # Any absorbers?
            if (predictions[jidx]['num_dlas'])+ (predictions[jidx]['num_subdlas']) == 0:
                continue
            lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
            # DLAs, subDLAs
            get_s2n_for_absorbers(sightline, lam, predictions[jidx]['dlas'])
            get_s2n_for_absorbers(sightline, lam, predictions[jidx]['subdlas'])

        runtime = timeit.default_timer() - process_timer
        print("Processed {:d} of {:d} in {:0.0f}s - {:0.2f}s per sample".format(
            sightlines_processed_count + num_sightlines, len(ids), runtime, runtime/num_sightlines))
        sightlines_processed_count += num_sightlines
    # Write
    print("About to over-write your JSON file.  Continue at your own risk!")
    # Return new predictions
    return predictions





