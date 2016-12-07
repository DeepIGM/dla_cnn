import matplotlib

matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np
import os, urllib, math, json, timeit, multiprocessing, gc, sys, warnings, re, pickle, gzip, h5py, itertools, glob
from traceback import print_exc
# from classification_model import predictions_ann as predictions_ann_c1
from localize_model import predictions_ann as predictions_ann_c2, predictions_to_central_wavelength
# from density_model import predictions_ann as predictions_ann_r1
from Dataset import Dataset
from astropy.io import fits
from astropy.table import Table
from multiprocessing import Process, Value, Array, Pool
from data_model.Sightline import Sightline
from data_model.Dla import Dla
from data_model.Id_GENSAMPLES import Id_GENSAMPLES
from data_model.Id_DR12 import Id_DR12
import code, traceback, signal


# DLAs from the DR9 catalog range from 920 to 1214, adding 120 on the right for variance in ly-a
# the last number is the number of pixels in SDSS sightlines that span the range
# REST_RANGE = [920, 1334, 1614]
# REST_RANGE = [911, 1346, 1696]
REST_RANGE = [900, 1346, 1748]
cache = {}              # Cache for files and resources that should be opened once and kept open
# igmtables = {}
# igmsp = None
TF_DEVICE = os.getenv('TF_DEVICE', '')


# Used for debugging when the process isn't responding.
def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d={'_frame':frame}         # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message  = "Signal received : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    i.interact(message)


def listen():
    signal.signal(signal.SIGUSR1, debug)  # Register handler


def normalize(data1, z_qso, divide_median=False):
    # flux
    flux = data1['flux']
    loglam = data1['loglam']
    lam = 10. ** loglam
    lam_rest = lam / (1. + z_qso)

    # npoints = np.shape(flux)[0]

    # pixel mask
    # or_mask = data1['or_mask']
    # bad_index = or_mask > 0
    # flux[bad_index] = np.nan

    if divide_median:
        # normalization
        idx1 = np.logical_and(lam_rest > 1270., lam_rest < 1290.)
        flux1 = flux[idx1]
        flux_mdn = np.nanmedian(flux1)

        # idx2 = np.logical_and(lam_rest >= 961., lam_rest <= 1216.75)
        flux_norm = flux / flux_mdn
        # lam_norm = lam_rest[idx2]

        assert (flux_norm.dtype == np.float32)
        return flux_norm
    else:
        return flux

        # interpolation
        # lam_interp = np.arange(start=961., stop=1217, step=0.25, dtype=np.float32)
        # flux_interp = np.interp(lam_interp, lam_norm, flux_norm)


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
    if not cache.has_key(fs) or not cache.has_key(fs):
        print "Cache miss: [%s] and/or [%s] not found in cache" % (fs,json_datafile)
    f = cache[fs] = h5py.File(fs, "r") if not cache.has_key(fs) else cache[fs]
    j = cache[json_datafile] = json.load(open(json_datafile)) if not cache.has_key(json_datafile) else cache[json_datafile]

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
    z_qso = meta['headers'][sightline.id.ix]['zem']

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
            igmsp = cache['igmsp'] = IgmSpec() if not cache.has_key('igmsp') else cache['igmsp']
            mtbl = cache[table_name] = Table(igmsp.idb.hdf[table_name + "/meta"].value) \
                if not cache.has_key(table_name) else cache[table_name]

            print "Plate/Fiber: ", plate, fiber
            plate = int(plate)
            fiber = int(fiber)

            # Find plate/fiber
            imt = np.where((mtbl['PLATE'] == plate) & (mtbl['FIBER'] == fiber))[0]
            igmid = mtbl['IGM_ID'][imt]
            # print "imt, igmid", imt, igmid, type(imt), type(igmid), type(mtbl), np.shape(mtbl), "end-print"
            assert np.shape(igmid)[0] == 1, "Expected igmid to contain exactly 1 value, found %d" % np.shape(igmid)[0]

            raw_data = {}
            spec, meta = igmsp.idb.grab_spec([table_name], igmid)

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


def scan_flux_about_central_wavelength(flux_normalized, loglam,
                                       central_wavelength,
                                       num_samples,
                                       kernel=400, pos_sample_kernel_percent=0.3):
    samples_buffer = np.zeros((10000, kernel), dtype=np.float32)
    offsets_buffer = np.zeros((10000,), dtype=np.float32)
    buffer_count = 0

    lam = 10.0 ** loglam
    ix_central = np.nonzero(lam >= central_wavelength)[0][0]

    # Generate positive samples shifted left and right around the central wavelength
    position_from = ix_central - kernel * pos_sample_kernel_percent / 2
    position_to = position_from + kernel * pos_sample_kernel_percent
    stride = (position_to - position_from) / num_samples
    if(ix_central < 200):
        import pdb; pdb.set_trace()

    for position in range(0, num_samples):
        ix_center_shift = round(position_from + stride * position)
        ix_from = int(ix_center_shift - kernel / 2)
        ix_to = int(ix_from + kernel)

        samples_buffer[buffer_count, :] = flux_normalized[ix_from:ix_to]
        offsets_buffer[buffer_count] = ix_center_shift - ix_central

        buffer_count += 1

    # return samples_buffer[0:buffer_count, :]
    return samples_buffer[0:buffer_count, :], offsets_buffer[0:buffer_count]


# def percent_nan_in_dla_range(flux, loglam, z_qso):
#     lam, lam_rest, ix_dla_range = get_lam_data(loglam, z_qso, REST_RANGE)
#     return float(np.sum(np.isnan(flux[ix_dla_range]))) / float(np.sum(ix_dla_range))


# def prepare_density_regression_train_test(kernel=400, pos_sample_kernel_percent=0.2, n_samples=80,
#                                           train_keys_csv="../data/dr9_train_set.csv",
#                                           test_keys_csv="../data/dr9_test_set.csv",
#                                           train_save_file="../data/densitydata_train.npy",
#                                           test_save_file="../data/densitydata_test.npy"):
#     dr9_train = np.genfromtxt(train_keys_csv, delimiter=',')
#     dr9_test = np.genfromtxt(test_keys_csv, delimiter=',')
#
#     data_train = np.zeros((2000000, kernel + 8), np.float32)
#     data_test = np.zeros((2000000, kernel + 8), np.float32)
#     loc_train = 0
#     loc_test = 0
#
#     # Training set
#     for i in range(0, np.shape(dr9_train)[0]):
#         try:
#             data1, z_qso = read_fits_file(dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2])
#             flux_norm = normalize(data1, z_qso)
#
#             data_pos = scan_flux_about_central_wavelength(flux_norm, data1['loglam'], z_qso,
#                                                           dr9_train[i, 3], dr9_train[i, 4], n_samples,
#                                                           dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2], -1, -1,
#                                                           kernel=kernel,
#                                                           pos_sample_kernel_percent=pos_sample_kernel_percent)
#             data_train[loc_train:loc_train + np.shape(data_pos)[0], :] = data_pos
#             loc_train += np.shape(data_pos)[0]
#
#             print(loc_train, np.shape(data_pos), np.shape(data_train))
#         except Exception as e:
#             print("Error ecountered on sample: ", dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2])
#             print_exc()
#             raise e
#
#     # Test set
#     for i in range(0, np.shape(dr9_test)[0]):
#         try:
#             data1, z_qso = read_fits_file(dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2])
#             flux_norm = normalize(data1, z_qso)
#
#             data_pos = scan_flux_about_central_wavelength(flux_norm, data1['loglam'], z_qso,
#                                                           dr9_test[i, 3], dr9_test[i, 4], n_samples,
#                                                           dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2], -1, -1,
#                                                           kernel=kernel,
#                                                           pos_sample_kernel_percent=pos_sample_kernel_percent)
#             data_test[loc_test:loc_test + np.shape(data_pos)[0], :] = data_pos
#             loc_test += np.shape(data_pos)[0]
#
#             percent_nan = percent_nan_in_dla_range(flux_norm, data1['loglam'], z_qso)
#             if percent_nan > 0.25:
#                 print("%0.3f,%d,%d,%d,%f,%f" %
#                       (percent_nan, dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2], dr9_test[i, 3], dr9_test[i, 4]))
#
#             print(loc_test, np.shape(data_pos), np.shape(data_test), 'nans', np.sum(np.isnan(data_pos[:-8])))
#         except Exception as e:
#             print("Error ecountered on sample: ", dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2])
#             print_exc()
#             raise e
#
#     np.save(train_save_file, data_train[0:loc_train, :])
#     np.save(test_save_file, data_test[0:loc_test, :])
#     return DataSet(data_train[0:loc_train, :]), DataSet(data_test[0:loc_test, :])


# Get the flux from the raw data normalized, truncated, and padded for classification
def get_raw_data_for_classification(data1, z_qso, label=-1, central_wavelength=-1, col_density=-1,
                                    plate=-1, mjd=-1, fiber=-1, ra=-1, dec=-1):
    # Clean up the flux data
    flux_norm = normalize(data1, z_qso)

    # lam = 10.0 ** data1['loglam']
    # lam_rest = lam / (1 + z_qso)
    # ix_rest_range = np.logical_and(lam_rest >= REST_RANGE[0], lam_rest <= REST_RANGE[1])
    lam, lam_rest, ix_rest_range = get_lam_data(data1['loglam'], z_qso, REST_RANGE)

    # print("ix_rest_range in get_raw_data_for_classification: ", np.sum(ix_rest_range))
    npy = np.zeros((1, REST_RANGE[2] + 8), dtype=np.float32)
    npy[0, :REST_RANGE[2]] = np.pad(flux_norm[ix_rest_range], (max(0, REST_RANGE[2] - np.sum(ix_rest_range)), 0), 'constant')
    npy[0, -1] = label
    npy[0, -2] = central_wavelength
    npy[0, -3] = col_density
    npy[0, -4] = plate
    npy[0, -5] = mjd
    npy[0, -6] = fiber
    npy[0, -7] = ra
    npy[0, -8] = dec

    return npy


# def prepare_classification_training_set(train_csv_file="../data/classification_train.csv",
#                                         test_dla_csv_file="../data/classification_test_dla.csv",
#                                         test_non_dla_csv_file="../data/classification_test_non_dla.csv",
#                                         save_train_file="../data/classification_train.npy",
#                                         save_test_dla_file="../data/classification_test_dla.npy",
#                                         save_test_non_dla_file="../data/classification_test_non_dla.npy"):
#     for (csv_file, save_file) in zip([train_csv_file, test_dla_csv_file, test_non_dla_csv_file],
#                                      [save_train_file, save_test_dla_file, save_test_non_dla_file]):
#         csv = np.genfromtxt(csv_file, delimiter=',')
#         # -3 labels mean sub-dla, count them as non-dla (0) in this pipeline
#         csv[csv[:, 4] == -3, 4] = 0
#         # Remove samples who's label is not 0 or 1 from the set
#         csv = csv[csv[:, 4] >= 0, :]
#
#         npy = np.zeros((np.shape(csv)[0], REST_RANGE[2] + 8), dtype=np.float32)
#
#         for i in range(0, np.shape(csv)[0]):
#             if i % 200 == 0 or i == np.shape(csv)[0] - 1:
#                 print("Processed %d samples in file %s" % (i, csv_file))
#             (data1, z_qso) = read_fits_file(csv[i, 0], csv[i, 1], csv[i, 2])
#             flux_for_classification = get_raw_data_for_classification(data1, z_qso, label=csv[i, 4],
#                                                                       plate=csv[i, 0], mjd=csv[i, 1], fiber=csv[i, 2])
#
#             # Pad the array to the left with zero's if it is short (REST_RANGE[2] in length), e.g. the sightline
#             # doesn't reach 920A in the rest frame
#             npy[i, :] = flux_for_classification
#
#         print("Saving file %s" % save_file)
#         np.save(save_file, npy)

# Reads a generic Sightline object
def read_sightline(sightline):
    id = sightline.id

    if isinstance(id, Id_DR12, ):
        data1, z_qso = read_fits_file(id.plate, id.mjd, id.fiber)
        sightline.id.ra = data1['ra']
        sightline.id.dec = data1['dec']
        sightline.flux = data1['flux']
        sightline.loglam = data1['loglam']
        sightline.z_qso = z_qso
    elif isinstance(id, Id_GENSAMPLES):
        read_custom_hdf5(sightline)
    else:
        raise Exception("%s not implemented yet" % type(id).__name__)


def preprocess_data_from_dr9(kernel=400, stride=3, pos_sample_kernel_percent=0.3,
                           train_keys_csv="../data/dr9_train_set.csv",
                           test_keys_csv="../data/dr9_test_set.csv"):
    dr9_train = np.genfromtxt(train_keys_csv, delimiter=',')
    dr9_test = np.genfromtxt(test_keys_csv, delimiter=',')

    # Dedup TODO (there aren't any in dr9_train, so skipping for now)
    # dr9_train_keys = np.vstack({tuple(row) for row in dr9_train[:,0:3]})

    sightlines_train = [Sightline(Id_DR12(s[0],s[1],s[2]),[Dla(s[3],s[4])]) for s in dr9_train]
    sightlines_test  = [Sightline(Id_DR12(s[0],s[1],s[2]),[Dla(s[3],s[4])]) for s in dr9_test]

    prepare_localization_training_set(kernel, stride, pos_sample_kernel_percent,
                                      sightlines_train, sightlines_test)


# Length 1 for non array elements
def pseudolen(p):
    return len(p) if hasattr(p,'__len__') else 1


# Set save_file parameters to null to return the results and not write them to disk
def preprocess_gensample_from_single_hdf5_file(kernel=400, stride=3, pos_sample_kernel_percent=0.3, percent_test=0.8,
                                                    hdf5_datafile='../data/training_100.hdf5',
                                                    json_datafile='../data/training_100.json',
                                                    train_save_file = "../data/localize_train",
                                                    test_save_file="../data/localize_test"):
    with open(json_datafile, 'r') as fj:
        n = len(json.load(fj))
        n_train = int((1-percent_test)*n)
        sightlines_train = [Sightline(Id_GENSAMPLES(i, hdf5_datafile, json_datafile)) for i in range(0,n_train)]
        sightlines_test  = [Sightline(Id_GENSAMPLES(i, hdf5_datafile, json_datafile)) for i in range(n_train,n)]

    if train_save_file is not None:
        prepare_localization_training_set(kernel, stride, pos_sample_kernel_percent,
                                          sightlines_train, sightlines_test,
                                          train_save_file=train_save_file,
                                          test_save_file=test_save_file)

    return sightlines_train, sightlines_test


def preprocess_gensamples_from_multiple_hdf5(kernel=400, stride=3, pos_sample_kernel_percent=0.3,
                                             hdf5_train_datafile_glob='../data/gensample_hdf5_files/training_*_5000.hdf5',
                                             json_train_datafile_glob='../data/gensample_hdf5_files/training_*_5000.json',
                                             hdf5_test_datafile_glob='../data/gensample_hdf5_files/test_*_5000.hdf5',
                                             json_test_datafile_glob='../data/gensample_hdf5_files/test_*_5000.json',
                                             train_save_file = "../data/localize_train",
                                             test_save_file = "../data/localize_test"):
    expanded_train_datafiles = sorted(glob.glob(hdf5_train_datafile_glob))
    expanded_train_json = sorted(glob.glob(json_train_datafile_glob))
    expanded_test_datafiles = sorted(glob.glob(hdf5_test_datafile_glob))
    expanded_test_json = sorted(glob.glob(json_test_datafile_glob))

    sightlines_train = []
    sightlines_test = []

    for hdf5_datafile, json_datafile in zip(expanded_train_datafiles, expanded_train_json):
        print "Processing ", hdf5_datafile, ", ", json_datafile
        sightlines_train.extend(
            preprocess_gensample_from_single_hdf5_file(kernel, stride, pos_sample_kernel_percent, 0.0,
                                                       hdf5_datafile=hdf5_datafile, json_datafile=json_datafile,
                                                       train_save_file=None, test_save_file=None)[0])

    for hdf5_datafile, json_datafile in zip(expanded_test_datafiles, expanded_test_json):
        print "Processing ", hdf5_datafile, ", ", json_datafile
        sightlines_test.extend(
            preprocess_gensample_from_single_hdf5_file(kernel, stride, pos_sample_kernel_percent, 0.0,
                                                       hdf5_datafile=hdf5_datafile, json_datafile=json_datafile,
                                                       train_save_file=None, test_save_file=None)[0])

    prepare_localization_training_set(kernel, stride, pos_sample_kernel_percent,
                                      sightlines_train, sightlines_test,
                                      train_save_file=train_save_file,
                                      test_save_file=test_save_file)


def parallel_map(fun, params_tuple):
    num_cores = multiprocessing.cpu_count() - 2
    lengths = np.array([pseudolen(p) for p in params_tuple])
    max_len = max(lengths)
    assert np.all(np.logical_or(lengths == 1, lengths == max_len))   # All params the same length or 1
    # split_count = int(math.ceil(float(max_len) / num_cores))

    z = []
    for param in params_tuple:
        if pseudolen(param) > 1:
            z.append(np.array_split(param,num_cores))
        else:
            z.append(np.repeat(param, num_cores))

    # print "DEBUG> -------------------------------------------------------"
    # print num_cores
    p = Pool(num_cores)
    result = p.map(fun, zip(*z)) #todo p.map
    p.close()
    p.join()
    return result


# Read fits files and prepare data into numpy files with train/test splits
def prepare_localization_training_set(kernel, stride, pos_sample_kernel_percent,
                                      # train_keys_csv="../data/dr9_train_set.csv",
                                      # test_keys_csv="../data/dr9_test_set.csv",
                                      sightlines_train, sightlines_test,
                                      train_save_file="../data/localize_train",
                                      test_save_file="../data/localize_test"):
    # Training set
    # for i in range(0, np.shape(dr9_train)[0]):
    # Returns an array of data_train dictionaries
    fields = ['flux','labels_classifier','labels_offset','col_density','central_wavelength',
              'plate','mjd','fiber','ra','dec']

    # Training data
    data_train = {}
    data_train_split = parallel_map(parallel_process_scan,
                                    (sightlines_train, kernel, stride, pos_sample_kernel_percent))
    for key in fields:
        data_train[key] = np.concatenate([d[key] for d in data_train_split])
    for k in data_train.keys():
        data_train[k] = data_train[k]
    for k in data_train.keys():
        data_train[k] = data_train[k]
    save_dataset(train_save_file, data_train)

    # Same for test data if it exists
    if len(sightlines_test) > 0:
        data_test = {}
        data_test_split = parallel_map(parallel_process_scan,
                                       (sightlines_test, kernel, stride, pos_sample_kernel_percent))
        for key in fields:
            data_test[key] = np.concatenate([d[key] for d in data_test_split])
        for k in data_test.keys():
            data_test[k] = data_test[k]
        save_dataset(test_save_file, data_test)


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

def parallel_process_scan((sightlines_train, kernel, stride, pos_sample_kernel_percent)):
    buff_size = 4000000
    data_train = {}
    data_train['flux'] = np.zeros((buff_size, kernel), np.float32)
    data_train['labels_classifier'] = np.zeros((buff_size,), np.float32)
    data_train['labels_offset'] = np.zeros((buff_size,), np.float32)
    data_train['col_density'] = np.zeros((buff_size,), np.float32)
    data_train['col_density_std'] = np.zeros((buff_size,), np.float32)
    data_train['central_wavelength'] = np.zeros((buff_size,), np.float32)
    data_train['plate'] = np.zeros((buff_size,), np.float32)
    data_train['mjd'] = np.zeros((buff_size,), np.float32)
    data_train['fiber'] = np.zeros((buff_size,), np.float32)
    data_train['ra'] = np.zeros((buff_size,), np.float32)
    data_train['dec'] = np.zeros((buff_size,), np.float32)

    loc_train = 0
    debug_thresh = 50000
    debug_sightlinecount = 0
    debug_n = len(sightlines_train)

    for sightline in sightlines_train:
        debug_sightlinecount += 1
        try:
            # data1, z_qso = read_fits_file(dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2])
            read_sightline(sightline)
            # sightline = Sightline(sightline_orig.id)        # hacky workaround for memory issues

            # Validate this sightline, if it's bad, we can skip it. Necessary to work around a few bad generate samples
            if not validate_sightline(sightline):
                import pdb; pdb.set_trace()
                print "Warning: sightline %s failed validation, skipping it" % sightline
                continue

            data1, z_qso = sightline.get_legacy_data1_format()

            flux_norm = normalize(data1, z_qso)
            begin_loc_train = loc_train        # We'll get a variable number of samples, this tracks the total we got

            # handle mutliple DLAs
            for dla in sightline.dlas:
                # Get negative samples
                neg_flux, neg_offsets = \
                    scan_flux_sample(flux_norm, data1['loglam'], z_qso, dla.central_wavelength,
                                     exclude_positive_samples=True, kernel=kernel, stride=stride,
                                     pos_sample_kernel_percent=pos_sample_kernel_percent)
                f = loc_train                               # from
                t = loc_train + np.shape(neg_flux)[0]       # to
                ones = np.ones((t-f,), dtype=np.float32)
                data_train['flux'][f:t,:] = neg_flux
                data_train['labels_classifier'][f:t] = 0 * ones
                data_train['labels_offset'][f:t] = neg_offsets
                data_train['col_density'][f:t] = 0 * ones
                data_train['central_wavelength'][f:t] = -1 * ones
                loc_train += neg_flux.shape[0]

                # Get positive samples
                pos_flux, pos_offsets = \
                    scan_flux_about_central_wavelength(flux_norm, data1['loglam'],
                                                       dla.central_wavelength, neg_flux.shape[0],
                                                       kernel=kernel, pos_sample_kernel_percent=pos_sample_kernel_percent)
                f = loc_train                               # from
                t = loc_train + np.shape(pos_flux)[0]       # to
                ones = np.ones((t-f,), dtype=np.float32)
                data_train['flux'][f:t,:] = pos_flux
                data_train['labels_classifier'][f:t] = ones
                data_train['labels_offset'][f:t] = pos_offsets
                data_train['col_density'][f:t] = dla.col_density * ones
                data_train['central_wavelength'][f:t] = dla.central_wavelength * ones
                loc_train += pos_flux.shape[0]

                # Add meta data
                ones = np.ones((loc_train-begin_loc_train,), dtype=np.float32)
                data_train['plate'][begin_loc_train:loc_train] = ones * (sightline.id.plate if hasattr(sightline.id, 'plate') else 0)
                data_train['mjd'][begin_loc_train:loc_train] = ones * (sightline.id.mjd if hasattr(sightline.id, 'mjd') else 0)
                data_train['fiber'][begin_loc_train:loc_train] = ones * (sightline.id.mjd if hasattr(sightline.id, 'fiber') else 0)

            sightline.clear()
            if(loc_train > debug_thresh):
                print "Processed %d into buffer, completed %d of %d sightlines in this thread" % (debug_thresh, debug_sightlinecount, debug_n)
                debug_thresh += 50000
        except Exception as e:
            print "Error ecountered on sample: ", sightline
            print_exc()
            raise e

    print "Completed %d training samples in thread" % loc_train
    # Replace the variable length buffer with the actual number of samples generated
    for k in data_train.keys():
        data_train[k] = data_train[k][0:loc_train]

    return data_train

# Saves the flux and other data (flux separately because pickle can't handle large numpy matrices)
def save_dataset(save_file, data):
    print "Writing %s.npy to disk" % save_file
    np.save(save_file+".npy", data['flux'])
    data['flux'] = None
    print "Writing %s.pickle to disk" % save_file
    with gzip.GzipFile(filename=save_file+".pickle", mode='wb', compresslevel=2) as f:
        pickle.dump([data], f, protocol=-1)


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


def multiprocess_read_fits_data(csv):
    data1, z_qso = read_fits_file(csv[0], csv[1], csv[2])
    return (data1, z_qso)


def multiprocess_read_igmspec(id):
    data1, z_qso = read_igmspec(id[0], id[1], id[2], id[3])
    return (data1, z_qso)



# Run the full sequence of predictions on a CSV list of sightlines.
# load_catalog: defines the way the data will be read (by fits file or igm library)
#               possible values: DR12_FITS - read DR12 from raw fits files
#                                DR7_IGM - read DR7 from Igm library
# CSV Format:
#   DR12: plate,mjd,fiber,ra,dec    # ra and dec are optional and unimplemented currently for DR12
#   DR7:  plate,fiber,ra,dec
def process_catalog(csv_plate_mjd_fiber="../../boss_catalog.csv", kernel_size=400, load_catalog='DR12_FITS',
                    CHUNK_SIZE=500,
                    MODEL_CHECKPOINT_C1 = "../models/classification_model",
                    MODEL_CHECKPOINT_C2 = "../models/localize_model",
                    MODEL_CHECKPOINT_R1 = "../models/density_model" ):

    csv = np.genfromtxt(csv_plate_mjd_fiber, delimiter=',')
    print "CSV read complete"

    sightline_results = []  # array of map objects containing the classification, and an array of DLAs

    for i in range(0, np.shape(csv)[0], CHUNK_SIZE):
        report_timer = timeit.default_timer()

        # Batch read fits file or from Igm library
        process_timer = timeit.default_timer()
        print "Reading up to %d fits files with %d cores" % (CHUNK_SIZE, multiprocessing.cpu_count() - 2)
        p = Pool(multiprocessing.cpu_count() - 2)
        if load_catalog == 'DR12_FITS':
            data1_zqso_tuple = p.map(multiprocess_read_fits_data, map(tuple, csv[i:i + CHUNK_SIZE, 0:3]))
        elif load_catalog == 'DR7_IGM':
            data1_zqso_tuple = p.map(multiprocess_read_igmspec, map(tuple, csv[i:i + CHUNK_SIZE, 0:4]))
        else:
            raise Exception("Impossible to reach code bug")
        p.close()
        p.join()

        print "Spectrum/Fits read done in %0.1f" % (timeit.default_timer() - process_timer)

        # Set up buffers to store data
        buff_size = min(CHUNK_SIZE, len(data1_zqso_tuple))
        c1_buffer = np.zeros((buff_size, REST_RANGE[2] + 8), dtype=np.float32)
        loglam_buffer = np.zeros((buff_size, REST_RANGE[2]), dtype=np.float32)
        z_qso_buffer = np.zeros((buff_size,), dtype=np.float32)
        c2_buffer = np.zeros((REST_RANGE[2] * buff_size, kernel_size), dtype=np.float32)
        r1_buffer = []
        c2_count = 0

        #
        # Batch pre-process data sets into proper numpy formats for C1, C2, and R1
        #
        gc.disable()  # Work around garbage collection bug: https://goo.gl/8YMYPH
        for ix, (data1, z_qso) in zip(range(buff_size), data1_zqso_tuple):
            # print "Processing ", data1['plate'], data1['mjd'], data1['fiber']       # Debug find bad fits files 
            c1_data = get_raw_data_for_classification(data1, z_qso,
                                                      plate=data1['plate'], mjd=data1['mjd'], fiber=data1['fiber'],
                                                      ra=data1['ra'], dec=data1['dec'])
            # print "DEBUG> ", data1['plate'], data1['mjd'], data1['fiber']
            c2_data,c2_offset = scan_flux_sample(normalize(data1, z_qso), data1['loglam'], z_qso, -1,
                                       exclude_positive_samples=False, kernel=kernel_size, stride=1,
                                       pos_sample_kernel_percent=0.3)

            lam, lam_rest, ix_dla_range = get_lam_data(data1['loglam'], z_qso, REST_RANGE)

            c1_buffer[ix, :] = c1_data
            loglam_buffer[ix, :] = data1['loglam'][ix_dla_range]
            z_qso_buffer[ix] = z_qso
            r1_buffer.append(data1)  # necessary_for_scan_about_central_wavelength for DLAs for right/left
            c2_buffer[c2_count:c2_count + np.shape(c2_data)[0]] = c2_data
            c2_count += np.shape(c2_data)[0]
            assert (np.shape(c2_data)[0] == REST_RANGE[2])
        gc.enable()

        #
        # Process pipeline of models
        #
        print "Processing pipeline of models"
        loc_pred, loc_conf, peaks_data, density_data_flat = \
            process_pipeline_for_batch(c1_buffer, c2_buffer, z_qso_buffer, loglam_buffer, r1_buffer,
                                       MODEL_CHECKPOINT_C1, MODEL_CHECKPOINT_C2, MODEL_CHECKPOINT_R1, kernel_size)

        #
        # Process output for each sightline
        #
        num_sightlines = len(data1_zqso_tuple)
        assert num_sightlines * REST_RANGE[2] == density_data_flat.shape[0]
        for ix in range(num_sightlines):
            density_data = density_data_flat[ix*REST_RANGE[2]:(ix+1)*REST_RANGE[2]]
            # Store classification level data in results
            sightline = ({
                'plate': int(c1_buffer[ix, -4]), 'mjd': int(c1_buffer[ix, -5]), 'fiber': int(c1_buffer[ix, -6]),
                'ra':float(c1_buffer[ix, -7]), 'dec':float(c1_buffer[ix, -8]),
                # 'classification': "HAS_DLA" if c1_pred[ix] else "NO_DLA",
                # 'classification_confidence': int((abs(0.5 - c1_conf[ix]) + 0.5) * 100),
                'z_qso': float(z_qso_buffer[ix]),
                'num_dlas': len(peaks_data[ix][7]),
                'dlas': []
            })
            # print "DEBUG> ", 'fiber ', int(c1_buffer[ix, -4]), ' mjd ', int(c1_buffer[ix, -5]), ' fiber ', int(c1_buffer[ix, -6])

            # Loop through peaks
            (peaks, peaks_uncentered, smoothed_sample, ixs_left, ixs_right,
             offset_hist, offset_conv_sum, peaks_offset) = peaks_data[ix]
            for peak in peaks_offset:
                lam, lam_rest, ix_dla_range = get_lam_data(loglam_buffer[ix,:], z_qso_buffer[ix], REST_RANGE)
                peak_lam_rest = lam_rest[ix_dla_range][peak]
                peak_lam_spectrum = peak_lam_rest * (1 + z_qso_buffer[ix])

                mean_col_density_prediction = np.mean(density_data[peak-40:peak+40])
                std_col_density_prediction = np.std(density_data[peak-40:peak+40])
                # dla_counter += 80
                z_dla = float(peak_lam_spectrum) / 1215.67 - 1
                sightline['dlas'].append({
                    'rest': float(peak_lam_rest),
                    'spectrum': float(peak_lam_spectrum),
                    'z_dla':float(z_dla),
                    'dla_confidence': float(smoothed_sample[peak]),
                    'column_density': float(mean_col_density_prediction),
                    'std_column_density': float(std_col_density_prediction)
                })
            sightline_results.append(sightline)
        #
        # Process pdfs for each sightline
        #
        # assert dla_counter == density_pred.shape[1]
        print "Processing PDFs"
        num_cores = multiprocessing.cpu_count() - 1
        # print c1_count
        split_count = int(math.ceil(float(np.shape(c1_buffer)[0]) / num_cores))
        split = range(split_count, np.shape(c1_buffer)[0], split_count)
        split_flat = np.array(split) * loglam_buffer.shape[1]
        p = Pool(num_cores)

        # density_pred_per_dla_split = 80 * np.cumsum([len(pd[7]) for pd in peaks_data])

        # sum every set of density predictions per dla, this # will vary by the # of dla's per prediction
        # density_pred_per_core_split = [j[-1] for j in np.split(density_pred_per_dla_split, split)][:-1]
        z1 = np.split(c1_buffer, split)
        z2 = np.split(c2_buffer, split_flat)
        z3 = np.split(loglam_buffer, split)
        z4 = np.split(z_qso_buffer, split)
        # z5 = np.split(c1_pred, split)
        # z6 = np.split(c1_conf, split)
        z7 = np.array_split(peaks_data, split)
        z8 = np.split(np.array(np.split(loc_conf, np.shape(c1_buffer)[0])), split)
        z9 = np.split(density_data_flat, split_flat)
        z10= np.array_split(r1_buffer, split)
        z = zip(z1, z2, z3, z4, z7, z8, z9, z10)
        assert len(z1) == len(z2) == len(z3) == len(z4) == len(z7) == len(z8) == len(z9) == len(z10), \
            "Length of PDF parameters don't match: %d %d %d %d %d %d %d %d" % \
            (len(z1), len(z2), len(z3), len(z4), len(z7), len(z8), len(z9), len(z10))
        assert len(z) <= num_cores
        p.map(generate_pdfs, z)
        p.close()
        p.join()

        print "Processed %d sightlines for reporting on %d cores in %0.2fs" % \
              (num_sightlines, num_cores, timeit.default_timer() - report_timer)

        runtime = timeit.default_timer() - process_timer
        print "Processed %d of %d in %0.0fs - %0.2fs per sample" % \
              (i + CHUNK_SIZE, np.shape(csv)[0], runtime, runtime / CHUNK_SIZE)

    # Write JSON string
    with open("../tmp/pipeline_predictions.json", 'w') as outfile:
        json.dump(sightline_results, outfile, indent=4)


# Generates a set of PDF visuals for each sightline and predictions
def generate_pdfs((c1_buffer, c2_buffer, loglam_buffer, z_qso_buffer,
                  peaks_data, loc_conf, density_data, r1_buffer)):
    PLOT_LEFT_BUFFER = 50       # The number of pixels to plot left of the predicted sightline
    dlas_counter = 0
    assert len(density_data) == np.shape(c1_buffer)[0] * REST_RANGE[2]

    for i in range(np.shape(c1_buffer)[0]):
        filename = "../tmp/visuals/dla-spec-%04d-%05d-%04d.pdf" % (c1_buffer[i, -4], c1_buffer[i, -5], c1_buffer[i, -6])
        # print filename
        pp = PdfPages(filename)

        (peaks, peaks_uncentered, smoothed_sample, ixs_left, ixs_right,
            offset_hist, offset_conv_sum, peaks_offset) = peaks_data[i]
        lam, lam_rest, ix_dla_range = get_lam_data(loglam_buffer[i, :], z_qso_buffer[i], REST_RANGE)
        full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(r1_buffer[i]['loglam'], z_qso_buffer[i], REST_RANGE)
        xlim = [REST_RANGE[0]-PLOT_LEFT_BUFFER, lam_rest[-1]]
        y = r1_buffer[i]['flux']
        y_plot_range = np.mean(y[y > 0]) * 3.0
        ylim = [-2, y_plot_range]

        # n_dlas = len(peaks_data[i][0])
        # n_plots = n_dlas + 3
        n_dlas_offset = len(peaks_data[i][7])
        n_plots_offset = n_dlas_offset + 3
        axtxt = 0
        axsl = 1
        axloc = 2

        # Plot DLA range
        fig, ax = plt.subplots(n_plots_offset, figsize=(20, (3.75 * n_plots_offset) + (0.1 * n_dlas_offset) + 0.15), sharex=False)

        ax[axsl].set_xlabel("Rest frame sightline in region of interest for DLAs with z_qso = [%0.4f]" % z_qso_buffer[i])
        ax[axsl].set_ylabel("Flux")
        ax[axsl].set_ylim(ylim)
        ax[axsl].set_xlim(xlim)
        ax[axsl].plot(full_lam_rest, r1_buffer[i]['flux'], '-k')

        # Plot 0-line
        ax[axsl].axhline(0, color='grey')

        # Plot z_qso line over sightline
        ax[axsl].plot((1216, 1216), (ylim[0], ylim[1]), 'k--')

        # Plot localization
        ax[axloc].set_xlabel("DLA Localization confidence & localization prediction(s)")
        ax[axloc].set_ylabel("Confidence")
        ax[axloc].plot(lam_rest, loc_conf[i, :], color='deepskyblue')
        ax[axloc].set_ylim([0, 1])
        ax[axloc].set_xlim(xlim)

        # Classification results
        textresult = "Classified %d-%d-%d (%0.5f ra / %0.5f dec) with %d DLAs\n" \
            % (c1_buffer[i, -4], c1_buffer[i, -5], c1_buffer[i, -6], c1_buffer[i, -7], c1_buffer[i, -8], n_dlas_offset)

        # Plot localization histogram
        ax[axloc].scatter(lam_rest, offset_hist, s=6, color='orange')
        ax[axloc].plot(lam_rest, offset_conv_sum, color='green')

        # Plot '+' peak markers
        ax[axloc].plot(lam_rest[peaks_offset], np.minimum(1, offset_conv_sum[peaks_offset]), '+', mew=5, ms=10, color='green', alpha=1)

        for pltix, peak in zip(range(axloc+1, n_plots_offset), peaks_offset):
            # Sightline plot transparent marker boxes
            ax[axsl].fill_between(lam_rest[peak - 10:peak + 10], y_plot_range, -2, color='green', lw=0, alpha=0.1)
            ax[axsl].fill_between(lam_rest[peak - 30:peak + 30], y_plot_range, -2, color='green', lw=0, alpha=0.1)
            ax[axsl].fill_between(lam_rest[peak - 50:peak + 50], y_plot_range, -2, color='green', lw=0, alpha=0.1)
            ax[axsl].fill_between(lam_rest[peak - 70:peak + 70], y_plot_range, -2, color='green', lw=0, alpha=0.1)

            # Plot column density measures with bar plots
            density_pred_per_this_dla = density_data[i*REST_RANGE[2]:(i+1)*REST_RANGE[2]][peak-40:peak+40]
            assert len(density_data[i*REST_RANGE[2]:(i+1)*REST_RANGE[2]]) == REST_RANGE[2]
            dlas_counter += 1
            mean_col_density_prediction = float(np.mean(density_pred_per_this_dla))
            ax[pltix].bar(np.arange(0, density_pred_per_this_dla.shape[0]), density_pred_per_this_dla, 0.25)
            ax[pltix].set_xlabel("Individual Column Density estimates for peak @ %0.0fA, +/- 0.3 of mean. " %
                                 (lam_rest[peak]) +
                                 "Mean: %0.3f - Median: %0.3f - Stddev: %0.3f" %
                                 (mean_col_density_prediction, float(np.median(density_pred_per_this_dla)),
                                  float(np.std(density_pred_per_this_dla))))
            ax[pltix].set_ylim([mean_col_density_prediction - 0.3, mean_col_density_prediction + 0.3])
            ax[pltix].plot(np.arange(0, density_pred_per_this_dla.shape[0]),
                           np.ones((density_pred_per_this_dla.shape[0]), np.float32) * mean_col_density_prediction)

            # Add DLA to test result
            textresult += \
                " > DLA central wavelength at: %0.0fA rest / %0.0fA spectrum w/ confidence %0.2f, has Column Density: %0.3f\n" \
                % (lam_rest[peak], lam_rest[peak] * (1 + z_qso_buffer[i]), smoothed_sample[peak], mean_col_density_prediction)

            ax[axloc].legend(['DLA classifier', 'Localization', 'DLA peak', 'Localization histogram'],
                             bbox_to_anchor=(1.0, 1.1))


        # # Old V1 peaks, this will be removed later
        # for pltix, peak, peak_uncentered, ix_left, ix_right in \
        #         zip(range(axloc + 1, n_plots), peaks, peaks_uncentered, ixs_left, ixs_right):
        #
        #     # Plot smoothed line
        #     ax[axloc].plot(lam_rest, smoothed_sample, color='blue', alpha=0.9)
        #
        #     # Plot peak '+' markers
        #     ctrl_pts_height = (smoothed_sample[ix_left] + smoothed_sample[ix_right]) / 2
        #     ax[axloc].plot(lam_rest[peak_uncentered], loc_conf[i, peak_uncentered], '+', mew=3, ms=7, color='red', alpha=1)
        #     ax[axloc].plot(lam_rest[peak], smoothed_sample[peak], '+', mew=3, ms=7, color='blue', alpha=0.8)
        #     ax[axloc].plot(lam_rest[ix_left], ctrl_pts_height, '+', mew=3, ms=7, color='orange', alpha=1)
        #     ax[axloc].plot(lam_rest[ix_right], ctrl_pts_height, '+', mew=3, ms=7, color='orange', alpha=1)

        # Display text
        ax[axtxt].text(0, 0, textresult, family='monospace', fontsize='xx-large')
        ax[axtxt].get_xaxis().set_visible(False)
        ax[axtxt].get_yaxis().set_visible(False)
        ax[axtxt].set_frame_on(False)

        plt.tight_layout()
        pp.savefig(figure=fig)
        pp.close()
        plt.close(fig)

    # assert dlas_counter * 80 == np.shape(density_pred)[1], \
    #     "dlas_counter, shape density_pred[1]: %d %d" % (
    #     dlas_counter, np.shape(density_pred)[1])  # Sanity check for bugs


def process_pipeline_for_batch(c1_buffer, c2_buffer, z_qso_buffer, loglam_buffer, r1_buffer,
                               MODEL_CHECKPOINT_C1, MODEL_CHECKPOINT_C2, MODEL_CHECKPOINT_R1, kernel_size):
    # Classification & localization models - Load model hyperparameter file & Generate predictions
    # with open(MODEL_CHECKPOINT_C1 + "_hyperparams.json", 'r') as fp:
    #     hyperparameters_c1 = json.load(fp)
    with open(MODEL_CHECKPOINT_C2 + "_hyperparams.json", 'r') as fp:
        hyperparameters_c2 = json.load(fp)
    # with open(MODEL_CHECKPOINT_R1 + "_hyperparams.json", 'r') as fp:
    #     hyperparameters_r1 = json.load(fp)

    num_sightlines = c1_buffer.shape[0]

    #
    # Model Predictions performed in batch
    #
    # print "C1 predictions begin"
    # c1_pred, c1_conf = predictions_ann_c1(hyperparameters_c1, c1_buffer[:, :-8], c1_buffer[:, -1],
    #                                       MODEL_CHECKPOINT_C1, TF_DEVICE=TF_DEVICE)
    print "C2 predictions begin"
    dummy_labels = np.zeros((c2_buffer.shape[0]))
    loc_pred, loc_conf, offsets, density_data = predictions_ann_c2(hyperparameters_c2, c2_buffer[:, :],
                                                              dummy_labels, dummy_labels, dummy_labels,
                                                              MODEL_CHECKPOINT_C2, TF_DEVICE=TF_DEVICE)

    print "Predictions to central wavelength begin"
    predtocent_timer = timeit.default_timer()
    peaks_data = predictions_to_central_wavelength(loc_conf, offsets, num_sightlines, 50, 300)
    print "Complete predictions to central wavelength in %0.1f" % (timeit.default_timer() - predtocent_timer)

    # n_density_est = 80
    # dla_count = np.sum([len(peak_data[7]) for peak_data in peaks_data])
    # density_data = np.zeros((dla_count * n_density_est, kernel_size), dtype=np.float32)
    # r1_count = 0

    # Loop through peaks generating full set of density estimate data samples
    # print "Processing peaks data begin"
    # for ix, (v1_peaks, v1_peaks_uncentered, smoothed_sample, ixs_left, ixs_right,
    #          offset_hist, offset_conv_sum, peaks_offset) in zip(range(np.shape(c1_pred)[0]), peaks_data):
    #     for peak in peaks_offset:
    #         z_qso = z_qso_buffer[ix]
    #         lam, lam_rest, ix_dla_range = get_lam_data(loglam_buffer[ix, :], z_qso, REST_RANGE)
    #         peak_lam_rest = lam_rest[ix_dla_range][peak]
    #
    #         density_data[r1_count:r1_count + n_density_est, :], ignore = \
    #             scan_flux_about_central_wavelength(r1_buffer[ix]['flux'], r1_buffer[ix]['loglam'], z_qso_buffer[ix],
    #                                                peak_lam_rest * (1 + z_qso), 0, n_density_est, 0, 0, 0, -1, -1,
    #                                                kernel_size, 0.2)
    #         r1_count += n_density_est

    # Column density model predictions
    # print "R1 predictions begin"
    # density_pred = predictions_ann_r1(hyperparameters_r1, density_data[:, :], c1_buffer[-1],
    #                                   MODEL_CHECKPOINT_R1, TF_DEVICE=TF_DEVICE)

    # Returns:
    #   c1_pred - Classifier 1 predictions (num_samples, )
    #   c1_conf - Classifer 1 confidence (num_samples, )
    #   loc_pred - location (classifer 2) predictions (num_samples, )
    #   loc_conf - location (classifer 2) confidences (num_samples, )
    #   peaks_data - peaks info per DLAs, list of tuples [
    #                   (peaks_centered, peaks_uncentered, smoothed_sample, ixs_left, ixs_right),
    #                   (peaks_centered, peaks_uncentered, smoothed_sample, ixs_left, ixs_right),
    #                   ... ]
    #   density_data - density predictions, 80 per DLA (num_dlas, 80)
    return loc_pred, loc_conf, peaks_data, density_data



listen()    # Adds a user signal listener for debugging purposes.