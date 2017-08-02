""" Module for loading spectra, either fake or real
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np
import os, urllib, math, json, timeit, multiprocessing, gc, sys, warnings, re, pickle, gzip, h5py, itertools, glob, time
from traceback import print_exc
import pdb

from pkg_resources import resource_filename

from astropy.io import fits
from astropy.table import Table
from multiprocessing import Process, Value, Array, Pool
from dla_cnn.data_model.Sightline import Sightline
from dla_cnn.data_model.Dla import Dla
from dla_cnn.data_model.Id_GENSAMPLES import Id_GENSAMPLES
from dla_cnn.data_model.Id_DR12 import Id_DR12
from dla_cnn.data_model.Id_DR7 import Id_DR7
from dla_cnn.data_model.Prediction import Prediction
from dla_cnn.data_model.DataMarker import Marker
import code, traceback, threading
from dla_cnn.localize_model import predictions_ann as predictions_ann_c2
import scipy.signal as signal
from scipy.spatial.distance import cdist
from scipy.signal import medfilt, find_peaks_cwt
from scipy.stats import chisquare
from scipy.optimize import minimize
from operator import itemgetter, attrgetter, methodcaller
from dla_cnn.Timer import Timer
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import astropy.units as u
from linetools.spectralline import AbsLine
from linetools.spectra import io as lsio
from linetools.analysis import voigt as lav
from linetools.analysis.voigt import voigt_from_abslines
from astropy.io.fits.hdu.compressed import compression

# Raise warnings to errors for debugging
import warnings
warnings.filterwarnings('error')


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
    lam, flux, _, _ = f['data'][ix]

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
    assert np.all(np.isfinite(lam) & np.isfinite(flux))

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
    (loglam_padded, flux_padded) = pad_loglam_flux(loglam, flux, z_qso)
    assert(np.all(np.logical_and(np.isfinite(loglam_padded), np.isfinite(flux_padded))))

    # sightline id
    sightline.id.sightlineid = j[str(ix)]['sl'] if j[str(ix)].has_key('sl') else -1

    sightline.dlas = []
    for dla_ix in range(0,int(j[str(ix)]['nDLA'])):
        central_wavelength = (1 + float(j[str(ix)][str(dla_ix)]['zabs'])) * 1215.67
        col_density = float(j[str(ix)][str(dla_ix)]['NHI'])
        sightline.dlas.append(Dla(central_wavelength, col_density))
    sightline.flux = flux_padded
    sightline.loglam = loglam_padded
    sightline.z_qso = z_qso

    if not validate_sightline(sightline):
        print("error validating sightline! bug! exiting")
        exit()

    return sightline


# Reads spectra out of IgmSpec library for DR7 (plate & fiber only)
def read_igmspec(plate, fiber, ra=-1, dec=-1, table_name='SDSS_DR7'):
    with open(os.devnull, 'w') as devnull:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Hack to avoid specdb spamming us with print statements
            stdout = sys.stdout
            sys.stdout = devnull

            from specdb.specdb import IgmSpec  # Custom package only used in this optional read function

            # global igmtables, igmsp
            global cache
            if ~cache.has_key(table_name):
                with lock:
                    if ~cache.has_key(table_name):
                        cache['igmsp'] = IgmSpec()
                        cache[table_name] = Table(cache['igmsp'].hdf[table_name + "/meta"].value)
            igmsp = cache['igmsp']
            mtbl = cache[table_name]

            print("Plate/Fiber: ", plate, fiber)
            plate = int(plate)
            fiber = int(fiber)

            # Find plate/fiber
            imt = np.where((mtbl['PLATE'] == plate) & (mtbl['FIBER'] == fiber))[0]
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
    pad_loglam = np.linspace(pad_loglam_lower, pad_loglam_upper,
                             max(0, (pad_loglam_upper - pad_loglam_lower + 0.0001) * 10000), dtype=np.float32)
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
    if isinstance(id, Id_DR12, ):
        data1, z_qso = read_fits_file(id.plate, id.mjd, id.fiber)
        sightline.id.ra = data1['ra']
        sightline.id.dec = data1['dec']
        sightline.flux = data1['flux']
        sightline.loglam = data1['loglam']
        sightline.z_qso = z_qso
    elif isinstance(id, Id_DR7):
        data1, z_qso = read_igmspec(id.plate, id.fiber, id.ra, id.dec)
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
    fluxes_matrix = np.vstack(map(lambda (f,r):f[r-kernelrangepx:r+kernelrangepx],
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


# Generates a catalog from plate/mjd/fiber from a CSV file
def process_catalog_dr7(csv_plate_mjd_fiber="../data/dr7_test_set.csv",
                        kernel_size=400, pfiber=None, make_pdf=False,
                        model_checkpoint=default_model,
                        output_dir="../tmp/visuals_dr7"):
    #csv = np.genfromtxt(csv_plate_mjd_fiber, delimiter=',')
    csv = Table.read(csv_plate_mjd_fiber)
    ids = [Id_DR7(c[0],c[1],c[2],c[3]) for c in csv]
    if pfiber is not None:
        plates = np.array([iid.plate for iid in ids])
        fibers = np.array([iid.fiber for iid in ids])
        imt = np.where((plates==pfiber[0]) & (fibers==pfiber[1]))[0]
        if len(imt) != 1:
            pdb.set_trace()
        else:
            ids = [ids[imt[0]]]
    process_catalog(ids, kernel_size, model_checkpoint, make_pdf=make_pdf,
                    CHUNK_SIZE=500, output_dir=output_dir)


def process_catalog_gensample(gensample_files_glob="../data/gensample_hdf5_files/test_mix_23559_10000.hdf5",
                              json_files_glob=     "../data/gensample_hdf5_files/test_mix_23559_10000.json",
                              kernel_size=400,
                              model_checkpoint=default_model,
                              output_dir="../tmp/visuals_gensample96451/"):
    expanded_datafiles = sorted(glob.glob(gensample_files_glob))
    expanded_json = sorted(glob.glob(json_files_glob))
    ids = []
    for hdf5_datafile, json_datafile in zip(expanded_datafiles, expanded_json):
        with open(json_datafile, 'r') as fj:
            n = len(json.load(fj))
            ids.extend([Id_GENSAMPLES(i, hdf5_datafile, json_datafile) for i in range(0, n)])
        process_catalog(ids, kernel_size, model_checkpoint, output_dir=output_dir)


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
                    make_pdf=False):
    num_cores = multiprocessing.cpu_count() - 1
    # num_cores = 24
    # p = None
    p = Pool(num_cores)  # a thread pool we'll reuse
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
            sightlines_batch.sort(key=lambda s: s.id.id_string())

        ##################################################################
        # Process output for each sightline
        ##################################################################
        assert num_sightlines * REST_RANGE[2] == density_data_flat.shape[0]
        for sightline in sightlines_batch:
            smoothed_sample = sightline.prediction.smoothed_loc_conf()

            dlas = []
            subdlas = []
            lybs = []

            # Loop through peaks which identify a DLA
            # (peaks, peaks_uncentered, smoothed_sample, ixs_left, ixs_right, offset_hist, offset_conv_sum, peaks_offset) \
            #     = peaks_data[ix]
            for peak in sightline.prediction.peaks_ixs:
                lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
                peak_lam_rest = lam_rest[ix_dla_range][peak]
                peak_lam_spectrum = peak_lam_rest * (1 + sightline.z_qso)

                # mean_col_density_prediction = np.mean(density_data[peak-40:peak+40])
                # std_col_density_prediction = np.std(density_data[peak-40:peak+40])
                z_dla = float(peak_lam_spectrum) / 1215.67 - 1
                _, mean_col_density_prediction, std_col_density_prediction, bias_correction = \
                    sightline.prediction.get_coldensity_for_peak(peak)

                absorber_type = "LYB" if sightline.is_lyb(peak) else "DLA" if mean_col_density_prediction >= 20.3 else "SUBDLA"
                dla_sub_lyb = lybs if absorber_type == "LYB" else dlas if absorber_type == "DLA" else subdlas

                # Should add S/N at peak
                dla_sub_lyb.append({
                    'rest': float(peak_lam_rest),
                    'spectrum': float(peak_lam_spectrum),
                    'z_dla':float(z_dla),
                    'dla_confidence': min(1.0,float(sightline.prediction.offset_conv_sum[peak])),
                    'column_density': float(mean_col_density_prediction),
                    'std_column_density': float(std_col_density_prediction),
                    'column_density_bias_adjust': float(bias_correction),
                    'type': absorber_type
                })

            # Store classification level data in results
            sightline_json = ({
                'id':           sightline.id.id_string(),
                'ra':           float(sightline.id.ra),
                'dec':          float(sightline.id.dec),
                'z_qso':        float(sightline.z_qso),
                'num_dlas':     len(dlas),
                'num_subdlas':  len(subdlas),
                'num_lyb':      len(lybs),
                'dlas':         dlas,
                'subdlas':      subdlas,
                'lyb':          lybs
            })

            sightline_results.append(sightline_json)

        ##################################################################
        # Process pdfs for each sightline
        ##################################################################
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # print "Processing PDFs"
        if make_pdf:
             p.map(generate_pdf, zip(sightlines_batch, itertools.repeat(output_dir)))  # TODO

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

# Estimate S/N at an absorber
def get_s2n_for_absorbers(sightline, lam, absorbers, nsamp=20):
    if len(absorbers) == 0:
        return
    # Loop on the DLAs
    for jj in range(len(absorbers)):
        # Find the peak
        isys = absorbers[jj]
        # Get the Voigt (to avoid it)
        voigt_flux, voigt_wave = generate_voigt_profile(isys['z_dla'], isys['column_density'], lam)
        # get peaks
        ixs_mypeaks = get_peaks_for_voigt_scaling(sightline, voigt_flux)
        if len(ixs_mypeaks) < 2:
            s2n = 1.  # KLUDGE
        else:
            # get indexes where voigt profile is between 0.2 and 0.95
            observed_values = sightline.flux[ixs_mypeaks]
            expected_values = voigt_flux[ixs_mypeaks]
            # Minimize scale variable using chi square measure for signal
            opt = minimize(lambda scale: chisquare(observed_values, expected_values * scale)[0], 1)
            opt_scale = opt.x[0]
            # Noise
            core = voigt_flux < 0.8
            rough_noise = np.median(sightline.sig[core])
            if rough_noise == 0:  # Occasional bad data in error array
                s2n = 0.1
            else:
                s2n = opt_scale/rough_noise
        isys['s2n'] = s2n
        '''  Another algorithm
        # Core
        core = np.where(voigt_flux < 0.8)[0]
        # Fluxes -- Take +/-nsamp away from core
        flux_for_stats = np.concatenate([sightline.flux[core[0]-nsamp:core[0]], sightline.flux[core[1]:core[1]+nsamp]])
        # Sort
        asrt = np.argsort(flux_for_stats)
        rough_signal = flux_for_stats[asrt][int(0.9*len(flux_for_stats))]
        rough_noise = np.median(sightline.sig[core])
        #
        s2n = rough_signal/rough_noise
        '''
    return


# Returns peaks used for voigt scaling, removes outlier and ensures enough points for good scaling
def get_peaks_for_voigt_scaling(sightline, voigt_flux):
    iteration_count = 0
    ixs_mypeaks_outliers_removed = []

    # Loop to try different find_peak values if we don't get enough peaks with one try
    while iteration_count < 10 and len(ixs_mypeaks_outliers_removed) < 5:
        peaks = np.array(find_peaks_cwt(sightline.flux, np.arange(1, 2+iteration_count)))
        ixs = np.where((voigt_flux > 0.2) & (voigt_flux < 0.95))
        ixs_mypeaks = np.intersect1d(ixs, peaks)

        # Remove any points > 1.5 standard deviations from the mean (poor mans outlier removal)
        peaks_mean = np.mean(sightline.flux[ixs_mypeaks]) if len(ixs_mypeaks)>0 else 0
        peaks_std = np.std(sightline.flux[ixs_mypeaks]) if len(ixs_mypeaks)>0 else 0

        ixs_mypeaks_outliers_removed = ixs_mypeaks[np.abs(sightline.flux[ixs_mypeaks] - peaks_mean) < (peaks_std * 1.5)]
        iteration_count += 1


    return ixs_mypeaks_outliers_removed


def generate_voigt_profile(dla_z, mean_col_density_prediction, full_lam):
    with open(os.devnull, 'w') as devnull:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Hack to avoid AbsLine spamming us with print statements
            stdout = sys.stdout
            sys.stdout = devnull

            abslin = AbsLine(1215.670 * 0.1 * u.nm, z=dla_z)
            abslin.attrib['N'] = 10 ** mean_col_density_prediction / u.cm ** 2  # log N
            abslin.attrib['b'] = 25. * u.km / u.s  # b
            # print dla_z, mean_col_density_prediction, full_lam, full_lam.shape
            # try:
            vmodel = voigt_from_abslines(full_lam.astype(np.float16) * u.AA, abslin, fwhm=3, debug=False)
            # except TypeError as e:
            #     import pdb; pdb.set_trace()
            voigt_flux = vmodel.data['flux'].data[0]
            voigt_wave = vmodel.data['wave'].data[0]
            # clear some bad values at beginning / end of voigt_flux
            voigt_flux[0:10] = 1
            voigt_flux[-10:len(voigt_flux) + 1] = 1

            sys.stdout = stdout

    return voigt_flux, voigt_wave

# Generates a PDF visuals for a sightline and predictions
def generate_pdf((sightline, path)):
    try:
        loc_conf = sightline.prediction.loc_conf
        peaks_offset = sightline.prediction.peaks_ixs
        offset_conv_sum = sightline.prediction.offset_conv_sum
        # smoothed_sample = sightline.prediction.smoothed_loc_conf()

        PLOT_LEFT_BUFFER = 50       # The number of pixels to plot left of the predicted sightline
        dlas_counter = 0

        filename = path + "/dla-spec-%s.pdf"%sightline.id.id_string()
        pp = PdfPages(filename)

        full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
        lam_rest = full_lam_rest[full_ix_dla_range]

        xlim = [REST_RANGE[0]-PLOT_LEFT_BUFFER, lam_rest[-1]]
        y = sightline.flux
        y_plot_range = np.mean(y[y > 0]) * 3.0
        ylim = [-2, y_plot_range]

        n_dlas = len(sightline.prediction.peaks_ixs)

        # Plot DLA range
        n_rows = 3 + (1 if n_dlas>0 else 0) + n_dlas
        fig = plt.figure(figsize=(20, (3.75 * (4+n_dlas)) + 0.15))
        axtxt = fig.add_subplot(n_rows,1,1)
        axsl = fig.add_subplot(n_rows,1,2)
        axloc = fig.add_subplot(n_rows,1,3)

        axsl.set_xlabel("Rest frame sightline in region of interest for DLAs with z_qso = [%0.4f]" % sightline.z_qso)
        axsl.set_ylabel("Flux")
        axsl.set_ylim(ylim)
        axsl.set_xlim(xlim)
        axsl.plot(full_lam_rest, sightline.flux, '-k')

        # Plot 0-line
        axsl.axhline(0, color='grey')

        # Plot z_qso line over sightline
        # axsl.plot((1216, 1216), (ylim[0], ylim[1]), 'k-', linewidth=2, color='grey', alpha=0.4)

        # Plot observer frame ticks
        axupper = axsl.twiny()
        axupper.set_xlim(xlim)
        xticks = np.array(axsl.get_xticks())[1:-1]
        axupper.set_xticks(xticks)
        axupper.set_xticklabels((xticks * (1 + sightline.z_qso)).astype(np.int32))

        # Sanity check
        if sightline.dlas and len(sightline.dlas) > 9:
            print("number of sightlines for {:s} is {:d}".format(
                sightline.id.id_string(), len(sightline.dlas)))

        # Plot given DLA markers over location plot
        for dla in sightline.dlas if sightline.dlas is not None else []:
            dla_rest = dla.central_wavelength / (1+sightline.z_qso)
            axsl.plot((dla_rest, dla_rest), (ylim[0], ylim[1]), 'g--')

        # Plot localization
        axloc.set_xlabel("DLA Localization confidence & localization prediction(s)")
        axloc.set_ylabel("Identification")
        axloc.plot(lam_rest, loc_conf, color='deepskyblue')
        axloc.set_ylim([0, 1])
        axloc.set_xlim(xlim)

        # Classification results
        textresult = "Classified %s (%0.5f ra / %0.5f dec) with %d DLAs/sub dlas/Ly-B\n" \
            % (sightline.id.id_string(), sightline.id.ra, sightline.id.dec, n_dlas)

        # Plot localization histogram
        axloc.scatter(lam_rest, sightline.prediction.offset_hist, s=6, color='orange')
        axloc.plot(lam_rest, sightline.prediction.offset_conv_sum, color='green')
        axloc.plot(lam_rest, sightline.prediction.smoothed_conv_sum(), color='yellow', linestyle='-', linewidth=0.25)

        # Plot '+' peak markers
        if len(peaks_offset) > 0:
            axloc.plot(lam_rest[peaks_offset], np.minimum(1, offset_conv_sum[peaks_offset]), '+', mew=5, ms=10, color='green', alpha=1)

        #
        # For loop over each DLA identified
        #
        for dlaix, peak in zip(range(0,n_dlas), peaks_offset):
            # Some calculations that will be used multiple times
            dla_z = lam_rest[peak] * (1 + sightline.z_qso) / 1215.67 - 1

            # Sightline plot transparent marker boxes
            axsl.fill_between(lam_rest[peak - 10:peak + 10], y_plot_range, -2, color='green', lw=0, alpha=0.1)
            axsl.fill_between(lam_rest[peak - 30:peak + 30], y_plot_range, -2, color='green', lw=0, alpha=0.1)
            axsl.fill_between(lam_rest[peak - 50:peak + 50], y_plot_range, -2, color='green', lw=0, alpha=0.1)
            axsl.fill_between(lam_rest[peak - 70:peak + 70], y_plot_range, -2, color='green', lw=0, alpha=0.1)

            # Plot column density measures with bar plots
            # density_pred_per_this_dla = sightline.prediction.density_data[peak-40:peak+40]
            dlas_counter += 1
            # mean_col_density_prediction = float(np.mean(density_pred_per_this_dla))
            density_pred_per_this_dla, mean_col_density_prediction, std_col_density_prediction, bias_correction = \
                sightline.prediction.get_coldensity_for_peak(peak)

            pltix = fig.add_subplot(n_rows, 1, 5+dlaix)
            pltix.bar(np.arange(0, density_pred_per_this_dla.shape[0]), density_pred_per_this_dla, 0.25)
            pltix.set_xlabel("Individual Column Density estimates for peak @ %0.0fA, +/- 0.3 of mean. Bias adjustment of %0.3f added. " %
                                 (lam_rest[peak], float(bias_correction)) +
                                 "Mean: %0.3f - Median: %0.3f - Stddev: %0.3f" %
                                 (mean_col_density_prediction, float(np.median(density_pred_per_this_dla)),
                                  float(std_col_density_prediction)))
            pltix.set_ylim([mean_col_density_prediction - 0.3, mean_col_density_prediction + 0.3])
            pltix.plot(np.arange(0, density_pred_per_this_dla.shape[0]),
                           np.ones((density_pred_per_this_dla.shape[0]), np.float32) * mean_col_density_prediction)
            pltix.set_ylabel("Column Density")

            # Add DLA to test result
            absorber_type = "Ly-b" if sightline.is_lyb(peak) else "DLA" if mean_col_density_prediction >= 20.3 else "sub dla"
            dla_text = \
                "%s at: %0.0fA rest / %0.0fA observed / %0.4f z, w/ confidence %0.2f, has Column Density: %0.3f" \
                % (absorber_type,
                   lam_rest[peak],
                   lam_rest[peak] * (1 + sightline.z_qso),
                   dla_z,
                   min(1.0, float(sightline.prediction.offset_conv_sum[peak])),
                   mean_col_density_prediction)
            textresult += " > " + dla_text + "\n"

            #
            # Plot DLA zoom view with voigt overlay
            #
            # Generate the voigt model using astropy, linetools, etc.
            voigt_flux, voigt_wave = generate_voigt_profile(dla_z, mean_col_density_prediction, full_lam)
            # get peaks
            ixs_mypeaks = get_peaks_for_voigt_scaling(sightline, voigt_flux)
            # get indexes where voigt profile is between 0.2 and 0.95
            observed_values = sightline.flux[ixs_mypeaks]
            expected_values = voigt_flux[ixs_mypeaks]
            # Minimize scale variable using chi square measure
            opt = minimize(lambda scale: chisquare(observed_values, expected_values * scale)[0], 1)
            opt_scale = opt.x[0]

            dla_min_text = \
                "%0.0fA rest / %0.0fA observed - NHI %0.3f" \
                % (lam_rest[peak],
                   lam_rest[peak] * (1 + sightline.z_qso),
                   mean_col_density_prediction)

            inax = fig.add_subplot(n_rows, n_dlas, n_dlas*3+dlaix+1)
            inax.plot(full_lam, sightline.flux, '-k', lw=1.2)
            inax.plot(full_lam[ixs_mypeaks], sightline.flux[ixs_mypeaks], '+', mew=5, ms=10, color='green', alpha=1)
            inax.plot(voigt_wave, voigt_flux * opt_scale, 'g--', lw=3.0)
            inax.set_ylim(ylim)
            # convert peak to index into full_lam range for plotting
            peak_full_lam = np.nonzero(np.cumsum(full_ix_dla_range) > peak)[0][0]
            inax.set_xlim([full_lam[peak_full_lam-150],full_lam[peak_full_lam+150]])
            inax.axhline(0, color='grey')

            #
            # Plot legend on location graph
            #
            axloc.legend(['DLA classifier', 'Localization', 'DLA peak', 'Localization histogram'],
                             bbox_to_anchor=(1.0, 1.05))


        # Display text
        axtxt.text(0, 0, textresult, family='monospace', fontsize='xx-large')
        axtxt.get_xaxis().set_visible(False)
        axtxt.get_yaxis().set_visible(False)
        axtxt.set_frame_on(False)

        fig.tight_layout()
        pp.savefig(figure=fig)
        pp.close()
        plt.close('all')

    except:
        print("Exception: ", traceback.format_exc())
