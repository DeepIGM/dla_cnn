import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np
import os, urllib, math, json, timeit, multiprocessing, gc, sys, warnings, re, pickle, gzip, h5py, itertools, glob, time
from traceback import print_exc
from astropy.io import fits
from astropy.table import Table
from multiprocessing import Process, Value, Array, Pool
from data_model.Sightline import Sightline
from data_model.Dla import Dla
from data_model.Id_GENSAMPLES import Id_GENSAMPLES
from data_model.Id_DR12 import Id_DR12
from data_model.Id_DR7 import Id_DR7
from data_model.Prediction import Prediction
import code, traceback, threading
from localize_model import predictions_ann as predictions_ann_c2
import scipy.signal as signal
from scipy.spatial.distance import cdist
from scipy.signal import medfilt, find_peaks_cwt
from scipy.stats import chisquare
from scipy.optimize import minimize
from operator import itemgetter, attrgetter, methodcaller
from Timer import Timer
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import astropy.units as u
from linetools.spectralline import AbsLine
from linetools.isgm.abscomponent import AbsComponent
from linetools.spectra import io as lsio
from linetools.analysis import voigt as lav
from linetools.analysis.voigt import voigt_from_abslines, voigt_from_components, voigt_wofz
from astropy.io.fits.hdu.compressed import compression


# DLAs from the DR9 catalog range from 920 to 1214, adding 120 on the right for variance in ly-a
# the last number is the number of pixels in SDSS sightlines that span the range
# REST_RANGE = [920, 1334, 1614]
# REST_RANGE = [911, 1346, 1696]
REST_RANGE = [900, 1346, 1748]
cache = {}              # Cache for files and resources that should be opened once and kept open
TF_DEVICE = os.getenv('TF_DEVICE', '')
lock = threading.Lock()



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
    meta = json.loads(f['meta'].value)
    # Two different runs named this key different things
    z_qso = meta['headers'][sightline.id.ix]['zem'] \
        if meta['headers'][sightline.id.ix].has_key('zem') else meta['headers'][sightline.id.ix]['zem_GROUP']

    # Pad loglam and flux_normalized to sufficiently below 920A rest that we don't have issues falling off the left
    (loglam_padded, flux_padded) = pad_loglam_flux(loglam, flux, z_qso)
    assert(np.all(np.logical_and(np.isfinite(loglam_padded), np.isfinite(flux_padded))))


    sightline.dlas = []
    for dla_ix in range(0,int(j[str(ix)]['nDLA'])):
        central_wavelength = (1 + float(j[str(ix)][str(dla_ix)]['zabs'])) * 1215.67
        col_density = float(j[str(ix)][str(dla_ix)]['NHI'])
        sightline.dlas.append(Dla(central_wavelength, col_density))
    sightline.flux = flux_padded
    sightline.loglam = loglam_padded
    sightline.z_qso = z_qso

    if not validate_sightline(sightline):
        import pdb; pdb.set_trace()

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

            print "Plate/Fiber: ", plate, fiber
            plate = int(plate)
            fiber = int(fiber)

            # Find plate/fiber
            imt = np.where((mtbl['PLATE'] == plate) & (mtbl['FIBER'] == fiber))[0]
            igmid = mtbl['IGM_ID'][imt]
            # print "imt, igmid", imt, igmid, type(imt), type(igmid), type(mtbl), np.shape(mtbl), "end-print"
            assert np.shape(igmid)[0] == 1, "Expected igmid to contain exactly 1 value, found %d" % np.shape(igmid)[0]

            raw_data = {}
            # spec, meta = igmsp.idb.grab_spec([table_name], igmid)
            spec, meta = igmsp.allspec_of_ID(igmid, groups=[table_name])

            z_qso = meta[0]['zem'][0]
            flux = np.array(spec[0].flux)
            loglam = np.log10(np.array(spec[0].wavelength))
            (loglam_padded, flux_padded) = pad_loglam_flux(loglam, flux, z_qso)
            # Sanity check that we're getting the log10 values
            assert np.all(loglam < 10), "Loglam values > 10, example: %f" % loglam[0]

            raw_data['flux'] = flux_padded
            raw_data['loglam'] = loglam_padded
            raw_data['plate'] = plate
            raw_data['mjd'] = 0
            raw_data['fiber'] = fiber
            raw_data['ra'] = ra
            raw_data['dec'] = dec
            assert np.shape(raw_data['flux']) == np.shape(raw_data['loglam'])
            sys.stdout = stdout

    return raw_data, z_qso


def pad_loglam_flux(loglam, flux, z_qso, kernel=1800):
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
    return loglam_padded, flux_padded


def scan_flux_sample(flux_normalized, loglam, z_qso, central_wavelength, #col_density, plate, mjd, fiber, ra, dec,
                     exclude_positive_samples=False, kernel=400, stride=5, pos_sample_kernel_percent=0.3):
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
            samples_buffer[buffer_count, :] = flux_normalized[position - kernel / 2:position - kernel / 2 + kernel]
            offsets_buffer[buffer_count] = 0
            buffer_count += 1
        elif not exclude_positive_samples:
            # Add a positive sample (is within pos_sample_kernel_percent of the central_wavelength)
            samples_buffer[buffer_count, :] = flux_normalized[position - kernel / 2:position - kernel / 2 + kernel]
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
                                               save_file = "../data/localize"):
    hdf5_datafile = datafile + ".hdf5"
    json_datafile = datafile + ".json"
    train_save_file = save_file + "_train" if percent_test > 0.0 else save_file
    test_save_file = save_file + "_test"

    with open(json_datafile, 'r') as fj:
        n = len(json.load(fj))
        n_train = int((1-percent_test)*n)
        ids_train = [Id_GENSAMPLES(i, hdf5_datafile, json_datafile) for i in range(0,n_train)]
        ids_test  = [Id_GENSAMPLES(i, hdf5_datafile, json_datafile) for i in range(n_train,n)]

    prepare_localization_training_set(kernel, stride, pos_sample_kernel_percent,
                                      ids_train, ids_test,
                                      train_save_file=train_save_file,
                                      test_save_file=test_save_file)


def preprocess_overlapping_dla_sightlines_from_gensample(kernel=400, stride=3, pos_sample_kernel_percent=0.3, percent_test=0.0,
                                               datafile='../data/gensample_hdf5_files/training*',
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
                                      test_save_file="../data/localize_test.npy"):
    num_cores = multiprocessing.cpu_count() - 1
    p = Pool(num_cores)       # a thread pool we'll reuse

    # Training data
    with Timer(disp="read_sightlines"):
        sightlines_train = p.map(read_sightline, ids_train)
    with Timer(disp="split_sightlines_into_samples"):
        data_split = p.map(split_sightline_into_samples, sightlines_train)
    with Timer(disp="select_samples_50p_pos_neg"):
        sample_masks = map(select_samples_50p_pos_neg, data_split)
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
    num_total = data[0].shape[0]
    ratio_neg = num_pos / num_neg

    pos_mask = classification == 1      # Take all positive samples

    neg_ixs_by_ratio = np.linspace(1,num_total-1,round(ratio_neg*num_total), dtype=np.int32) # get all samples by ratio
    neg_mask = np.zeros((num_total),dtype=np.bool) # create a 0 vector of negative samples
    neg_mask[neg_ixs_by_ratio] = True # set the vector to positives, selecting for the appropriate ratio across the whole sightline
    neg_mask[pos_mask] = False # remove previously positive samples from the set
    neg_mask[classification == -1] = False # remove border samples from the set, what remains is still in the right ratio

    return pos_mask | neg_mask


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
    print "Writing %s.npy to disk" % save_file
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

    # CLASSIFICATION
    # Start with all samples negative
    classification = np.zeros((REST_RANGE[2]), dtype=np.float32)
    # overlay samples that are too close to a known DLA, write these for all DLAs before overlaying positive sample 1's
    r = samplerangepx + kernelrangepx
    for ix_dla in ix_dlas:
        classification[ix_dla-r:ix_dla+r+1] = -1
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
    # classification is 1 / 0 / -1 for DLA/nonDLA/boarder
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
        if peak <= 40 or peak >= REST_RANGE[2]-40:
            smooth_conv_sum[max(0,peak-15):peak+15] = 0
            continue
        # move to the middle of the peak if there are multiple equal values
        ridge = 1
        while smooth_conv_sum[peak] == smooth_conv_sum[peak+ridge]:
            ridge += 1
        peak = peak + ridge/2
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
                        kernel_size=400,
                        model_checkpoint="../models/model_gensample_v4.4"):
    csv = np.genfromtxt(csv_plate_mjd_fiber, delimiter=',')
    ids = [Id_DR7(c[0],c[1],c[2],c[3]) for c in csv]
    process_catalog(ids, kernel_size, model_checkpoint, CHUNK_SIZE=500)


def process_catalog_gensample(gensample_files_glob="../data/gensample_hdf5_files/test_96451_5000.hdf5",
                              json_files_glob="../data/gensample_hdf5_files/test_96451_5000.json",
                              kernel_size=400,
                              model_checkpoint="../models/model_gensample_v2",
                              output_dir="../tmp/visuals/"):
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
                             model_checkpoint="../models/model_gensample_v4.4",
                             output_dir="../tmp/visuals/",
                             kernel_size=400):
    ids = []
    for f in glob.glob(fits_dir + "/*.fits"):
        match = re.match(r'.*-(\d+)-(\d+)-(\d+)\..*', f)
        if not match:
            print "Match failed on: ", f
            exit()
        ids.append(Id_DR12(int(match.group(1)),int(match.group(2)),int(match.group(3))))

    process_catalog(ids, kernel_size=kernel_size, model_path=model_checkpoint, output_dir=output_dir)


def process_catalog_csv_pmf(csv="../data/boss_catalog.csv",
                            model_checkpoint="../models/model_gensample_v4.4",
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
def process_catalog(ids, kernel_size, model_path="../models/model_gensample_v2",
                    CHUNK_SIZE=1000, output_dir="../tmp/visuals/"):
    num_cores = multiprocessing.cpu_count() - 1
    # num_cores = 24
    p = None
    sightlines_processed_count = 0

    sightline_results = []  # array of map objects containing the classification, and an array of DLAs
    ids.sort(key=methodcaller('id_string'))

    # We'll handle the full process in batches so as to not exceed memory constraints
    for ids_batch in np.array_split(ids, np.arange(CHUNK_SIZE,len(ids),CHUNK_SIZE)):
        # Workaround for segfaults occuring in matplotlib, kill multiprocess pool every iteration
        if p is not None:
            p.close()
            p.join()
            time.sleep(5)
        p = Pool(num_cores)  # a thread pool we'll reuse

        report_timer = timeit.default_timer()
        num_sightlines = len(ids_batch)

        # Batch read files
        process_timer = timeit.default_timer()
        print "Reading %d sightlines with %d cores" % (num_sightlines, num_cores)
        sightlines_batch = p.map(read_sightline, ids_batch)
        print "Spectrum/Fits read done in %0.1f" % (timeit.default_timer() - process_timer)

        ##################################################################
        # Process model
        ##################################################################
        print "Model predictions begin"
        fluxes = np.vstack([scan_flux_sample(s.flux, s.loglam, s.z_qso, -1, stride=1)[0] for s in sightlines_batch])
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
            # density_data = density_data_flat[ix*REST_RANGE[2]:(ix+1)*REST_RANGE[2]]
            # Store classification level data in results
            sightline_json = ({
                'id':       sightline.id.id_string(),
                'ra':       float(sightline.id.ra),
                'dec':      float(sightline.id.dec),
                'z_qso':    float(sightline.z_qso),
                'num_dlas': len(sightline.prediction.peaks_ixs),
                'dlas':     []
            })

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
                _, mean_col_density_prediction, std_col_density_prediction = \
                    sightline.prediction.get_coldensity_for_peak(peak)
                sightline_json['dlas'].append({
                    'rest': float(peak_lam_rest),
                    'spectrum': float(peak_lam_spectrum),
                    'z_dla':float(z_dla),
                    'dla_confidence': min(1.0,float(sightline.prediction.offset_conv_sum[peak])),
                    'column_density': float(mean_col_density_prediction),
                    'std_column_density': float(std_col_density_prediction)
                })
            sightline_results.append(sightline_json)

        ##################################################################
        # Process pdfs for each sightline
        ##################################################################
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print "Processing PDFs"
        #p.map(generate_pdf, zip(sightlines_batch, itertools.repeat(output_dir)))

        print "Processed %d sightlines for reporting on %d cores in %0.2fs" % \
              (num_sightlines, num_cores, timeit.default_timer() - report_timer)

        runtime = timeit.default_timer() - process_timer
        print "Processed %d of %d in %0.0fs - %0.2fs per sample" % \
              (sightlines_processed_count + num_sightlines, len(ids), runtime, runtime/num_sightlines)
        sightlines_processed_count += num_sightlines

    # Write JSON string
    # import pdb; pdb.set_trace()
    with open(output_dir + "/predictions.json", 'w') as outfile:
        json.dump(sightline_results, outfile, indent=4)


# Generates a set of PDF visuals for each sightline and predictions
def generate_pdf((sightline, path)):
    loc_conf = sightline.prediction.loc_conf
    peaks_offset = sightline.prediction.peaks_ixs
    offset_conv_sum = sightline.prediction.offset_conv_sum
    smoothed_sample = sightline.prediction.smoothed_loc_conf()

    PLOT_LEFT_BUFFER = 50       # The number of pixels to plot left of the predicted sightline
    dlas_counter = 0
    # assert len(density_data) == len(sightline.dlas) * REST_RANGE[2]

    filename = path + "/dla-spec-%s.pdf"%sightline.id.id_string()
    pp = PdfPages(filename)

    # (peaks, peaks_uncentered, smoothed_sample, ixs_left, ixs_right, offset_hist, offset_conv_sum, peaks_offset) \
    #     = peaks_data
    # lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    # full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(r1_buffer[i]['loglam'], z_qso_buffer[i], REST_RANGE)
    full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    lam_rest = full_lam_rest[full_ix_dla_range]
    lam = full_lam[full_ix_dla_range]

    xlim = [REST_RANGE[0]-PLOT_LEFT_BUFFER, lam_rest[-1]]
    # y = r1_buffer[i]['flux']
    y = sightline.flux
    y_plot_range = np.mean(y[y > 0]) * 3.0
    ylim = [-2, y_plot_range]

    # n_dlas = len(peaks_data[i][0])
    # n_plots = n_dlas + 3
    n_dlas_offset = len(sightline.prediction.peaks_ixs)
    n_plots_offset = n_dlas_offset + 4
    axtxt = 0
    axsl = 1
    axloc = 2
    axzm = 3

    # Plot DLA range
    fig, ax = plt.subplots(n_plots_offset, figsize=(20, (3.75 * n_plots_offset) + (0.1 * n_dlas_offset) + 0.15), sharex=False)#, dpi=900)  #todo issue here

    ax[axsl].set_xlabel("Rest frame sightline in region of interest for DLAs with z_qso = [%0.4f]" % sightline.z_qso)
    ax[axsl].set_ylabel("Flux")
    ax[axsl].set_ylim(ylim)
    ax[axsl].set_xlim(xlim)
    ax[axsl].plot(full_lam_rest, sightline.flux, '-k')

    # Plot 0-line
    ax[axsl].axhline(0, color='grey')

    # Plot z_qso line over sightline
    ax[axsl].plot((1216, 1216), (ylim[0], ylim[1]), 'k--', linewidth=2, color='grey')

    # Plot observer frame ticks
    axupper = ax[axsl].twiny()
    axupper.set_xlim(xlim)
    xticks = np.array(ax[axsl].get_xticks())[1:-1]
    axupper.set_xticks(xticks)
    axupper.set_xticklabels((xticks * (1 + sightline.z_qso)).astype(np.int32))

    # Plot given DLA markers over location plot
    for dla in sightline.dlas if sightline.dlas is not None else []:
        dla_rest = dla.central_wavelength / (1+sightline.z_qso)
        ax[axsl].plot((dla_rest, dla_rest), (ylim[0], ylim[1]), 'g--')

    # Plot localization
    ax[axloc].set_xlabel("DLA Localization confidence & localization prediction(s)")
    ax[axloc].set_ylabel("Identification")
    ax[axloc].plot(lam_rest, loc_conf, color='deepskyblue')
    ax[axloc].set_ylim([0, 1])
    ax[axloc].set_xlim(xlim)

    # Classification results
    textresult = "Classified %s (%0.5f ra / %0.5f dec) with %d DLAs\n" \
        % (sightline.id.id_string(), sightline.id.ra, sightline.id.dec, n_dlas_offset)

    # Plot localization histogram
    ax[axloc].scatter(lam_rest, sightline.prediction.offset_hist, s=6, color='orange')
    ax[axloc].plot(lam_rest, sightline.prediction.offset_conv_sum, color='green')
    ax[axloc].plot(lam_rest, sightline.prediction.smoothed_conv_sum(), color='yellow', linestyle='-', linewidth=0.25)

    # Plot '+' peak markers
    if len(peaks_offset) > 0:
        ax[axloc].plot(lam_rest[peaks_offset], np.minimum(1, offset_conv_sum[peaks_offset]), '+', mew=5, ms=10, color='green', alpha=1)

    #
    # For loop over each DLA identified
    #
    for dlaix, pltix, peak in zip(range(0,n_dlas_offset), range(axzm+1, n_plots_offset), peaks_offset):
        # Some calculations that will be used multiple times
        dla_z = lam_rest[peak] * (1 + sightline.z_qso) / 1215.67 - 1

        # Sightline plot transparent marker boxes
        ax[axsl].fill_between(lam_rest[peak - 10:peak + 10], y_plot_range, -2, color='green', lw=0, alpha=0.1)
        ax[axsl].fill_between(lam_rest[peak - 30:peak + 30], y_plot_range, -2, color='green', lw=0, alpha=0.1)
        ax[axsl].fill_between(lam_rest[peak - 50:peak + 50], y_plot_range, -2, color='green', lw=0, alpha=0.1)
        ax[axsl].fill_between(lam_rest[peak - 70:peak + 70], y_plot_range, -2, color='green', lw=0, alpha=0.1)

        # Plot column density measures with bar plots
        # density_pred_per_this_dla = sightline.prediction.density_data[peak-40:peak+40]
        dlas_counter += 1
        # mean_col_density_prediction = float(np.mean(density_pred_per_this_dla))
        density_pred_per_this_dla, mean_col_density_prediction, std_col_density_prediction = \
            sightline.prediction.get_coldensity_for_peak(peak)

        ax[pltix].bar(np.arange(0, density_pred_per_this_dla.shape[0]), density_pred_per_this_dla, 0.25)
        ax[pltix].set_xlabel("Individual Column Density estimates for peak @ %0.0fA, +/- 0.3 of mean. " %
                             (lam_rest[peak]) +
                             "Mean: %0.3f - Median: %0.3f - Stddev: %0.3f" %
                             (mean_col_density_prediction, float(np.median(density_pred_per_this_dla)),
                              float(std_col_density_prediction)))
        ax[pltix].set_ylim([mean_col_density_prediction - 0.3, mean_col_density_prediction + 0.3])
        ax[pltix].plot(np.arange(0, density_pred_per_this_dla.shape[0]),
                       np.ones((density_pred_per_this_dla.shape[0]), np.float32) * mean_col_density_prediction)
        ax[pltix].set_ylabel("Column Density")

        # Add DLA to test result
        dla_text = \
            "DLA at: %0.0fA rest / %0.0fA observed / %0.4f z, w/ confidence %0.2f, has Column Density: %0.3f" \
            % (lam_rest[peak],
               lam_rest[peak] * (1 + sightline.z_qso),
               dla_z,
               min(1.0, float(sightline.prediction.offset_conv_sum[peak])),
               mean_col_density_prediction)
        textresult += " > " + dla_text + "\n"

        #
        # Plot DLA zoom view with voigt overlay
        #
        # Generate the voigt model using astropy, linetools, etc.
        abslin = AbsLine(1215.670 * 0.1 * u.nm, z=dla_z)
        abslin.attrib['N'] = 10 ** mean_col_density_prediction / u.cm ** 2  # log N
        abslin.attrib['b'] = 25. * u.km / u.s  # b
        print "DEBUG> before vmodel ", os.getpid()
        vmodel = voigt_from_abslines(full_lam*u.AA, abslin, fwhm=3, debug=True)
        print "       after vmodel ", os.getpid()
        voigt_flux = vmodel.data['flux'].data[0]
        voigt_wave = vmodel.data['wave'].data[0]
        # clear some bad values at beginning / end of voigt_flux
        voigt_flux[0:10] = 1
        voigt_flux[-10:len(voigt_flux) + 1] = 1
        # get peaks
        peaks = np.array(find_peaks_cwt(sightline.flux, np.arange(1, 3)))
        # get indexes where voigt profile is between 0.2 and 0.9
        ixs = np.where((voigt_flux > 0.2) & (voigt_flux < 0.9))
        ixs_mypeaks = np.intersect1d(ixs, peaks)
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

        ax[axzm].axis('off')
        inax = inset_axes(ax[axzm], width=str(int(1.0/n_dlas_offset*100)-3)+"%", height="100%", loc=axzm+dlaix)
        inax.set_xlabel(dla_min_text)
        assert len(full_lam) == len(sightline.flux)
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
        ax[axloc].legend(['DLA classifier', 'Localization', 'DLA peak', 'Localization histogram'],
                         bbox_to_anchor=(1.0, 1.05))


    # Display text
    ax[axtxt].text(0, 0, textresult, family='monospace', fontsize='xx-large')
    ax[axtxt].get_xaxis().set_visible(False)
    ax[axtxt].get_yaxis().set_visible(False)
    ax[axtxt].set_frame_on(False)

    plt.tight_layout()
    pp.savefig(figure=fig)
    pp.close()
    plt.close(fig)


