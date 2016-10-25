import matplotlib

matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np
import os, urllib, math, json, timeit, multiprocessing, gc, sys, warnings, re
from traceback import print_exc
from classification_model import predictions_ann as predictions_ann_c1
from localize_model import predictions_ann as predictions_ann_c2, predictions_to_central_wavelength
from density_model import predictions_ann as predictions_ann_r1
from DataSet import DataSet
from astropy.io import fits
from astropy.table import Table
from multiprocessing import Process, Value, Array, Pool
import code, traceback, signal

# DLAs from the DR9 catalog range from 920 to 1214, adding 120 on the right for errors in ly-a
# the last number is the number of pixels in SDSS sightlines that span the range
REST_RANGE = [920, 1334, 1614]
# REST_RANGE = [800, 1334, 2221]


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
    fits_file = fits.open(fits_filename)
    data1 = fits_file[1].data
    z_qso = fits_file[3].data['LINEZ'][0]
    raw_data = {}

    # Pad loglam and flux_normalized to sufficiently below 920A rest that we don't have issues falling off the left
    (loglam_padded, flux_padded) = pad_loglam_flux(data1['loglam'], data1['flux'], z_qso, 800)
    raw_data['flux'] = flux_padded
    raw_data['loglam'] = loglam_padded
    raw_data['plate'] = fits_file[2].data['PLATE']
    raw_data['mjd'] = fits_file[2].data['MJD']
    raw_data['fiber'] = fits_file[2].data['FIBERID']
    raw_data['ra'] = fits_file[2].data['RA']
    raw_data['dec'] = fits_file[2].data['DEC']

    return raw_data, z_qso


# Reads spectra out of IgmSpec library for DR7 (plate & fiber only)
def read_igmspec(plate, fiber, ra=-1, dec=-1, table_name='SDSS_DR7'):
    stdout = sys.stdout
    with open(os.devnull, 'w') as devnull:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            sys.stdout = devnull
            print "Plate/Fiber: ", plate, fiber
            from specdb.specdb import IgmSpec  # Custom package only used in this optional read function
            plate = int(plate)
            fiber = int(fiber)

            igmsp = IgmSpec()
            mtbl = Table(igmsp.idb.hdf[table_name + "/meta"].value)

            # Find plate/fiber
            imt = np.where((mtbl['PLATE'] == plate) & (mtbl['FIBER'] == fiber))[0]
            igmid = mtbl['IGM_ID'][imt]
            # print "imt, igmid", imt, igmid, type(imt), type(igmid), type(mtbl), np.shape(mtbl), "end-print"
            assert np.shape(igmid)[0] == 1, "Expected igmid to contain exactly 1 value, found %d" % np.shape(igmid)[0]

            raw_data = {}
            spec, meta = igmsp.idb.grab_spec(['SDSS_DR7'], igmid)

            z_qso = meta[0]['zem'][0]
            flux = np.array(spec[0].flux)
            loglam = np.log10(np.array(spec[0].wavelength))
            (loglam_padded, flux_padded) = pad_loglam_flux(loglam, flux, z_qso, 800)
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


def pad_loglam_flux(loglam, flux, z_qso, kernel):
    kernel = 1200    # Overriding left padding to increase it
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


def scan_flux_sample(flux_normalized, loglam, z_qso, central_wavelength, col_density, plate, mjd, fiber, ra, dec,
                     exclude_positive_samples=False, kernel=400, stride=5, pos_sample_kernel_percent=0.3):
    # Split from rest frame 920A to 1214A (the range of DLAs in DR9 catalog)
    # pos_sample_kernel_percent is the percent of the kernel where positive samples can be found
    # e.g. the central wavelength is within this percentage of the center of the kernel window

    # Pre allocate space for generating samples
    samples_buffer = np.zeros((10000, kernel + 8), dtype=np.float32)
    buffer_count = 0

    # # Pad loglam and flux_normalized with kernel/2 zeros so scanning can start at 920A or higher
    # print('before', np.shape(loglam), np.shape(flux_normalized))
    # (loglam, flux_normalized) = pad_loglam_flux(loglam, flux_normalized, z_qso, kernel)
    # print('after', np.shape(loglam), np.shape(flux_normalized))

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
            samples_buffer[buffer_count, :-8] = flux_normalized[position - kernel / 2:position - kernel / 2 + kernel]
            samples_buffer[buffer_count, -1] = 0
            samples_buffer[buffer_count, -2] = 0
            samples_buffer[buffer_count, -3] = 0
            samples_buffer[buffer_count, -4] = plate
            samples_buffer[buffer_count, -5] = mjd
            samples_buffer[buffer_count, -6] = fiber
            samples_buffer[buffer_count, -7] = ra
            samples_buffer[buffer_count, -8] = dec
            buffer_count += 1
        elif not exclude_positive_samples:
            # Add a positive sample (is within pos_sample_kernel_percent of the central_wavelength)
            samples_buffer[buffer_count, :-8] = flux_normalized[position - kernel / 2:position - kernel / 2 + kernel]
            samples_buffer[buffer_count, -1] = 1
            samples_buffer[buffer_count, -2] = col_density
            samples_buffer[buffer_count, -3] = central_wavelength
            samples_buffer[buffer_count, -4] = plate
            samples_buffer[buffer_count, -5] = mjd
            samples_buffer[buffer_count, -6] = fiber
            samples_buffer[buffer_count, -7] = ra
            samples_buffer[buffer_count, -8] = dec
            buffer_count += 1

    return samples_buffer[0:buffer_count, :]


def scan_flux_about_central_wavelength(flux_normalized, loglam, z_qso, central_wavelength, col_density, num_samples,
                                       plate, mjd, fiber, ra, dec, kernel=400, pos_sample_kernel_percent=0.3):
    samples_buffer = np.zeros((10000, kernel + 8), dtype=np.float32)
    buffer_count = 0

    # # Pad loglam and flux_normalized with kernel/2 zeros so scanning can start at 920A or higher
    # (loglam, flux_normalized) = pad_loglam_flux(loglam, flux_normalized, z_qso, kernel)

    lam = 10.0 ** loglam
    ix_central = np.nonzero(lam >= central_wavelength)[0][0]

    # Generate positive samples shifted left and right around the central wavelength
    positive_position_from = ix_central - kernel * pos_sample_kernel_percent / 2
    positive_position_to = positive_position_from + kernel * pos_sample_kernel_percent
    positive_stride = (positive_position_to - positive_position_from) / num_samples

    for position in range(0, num_samples):
        ix_center_shift = round(positive_position_from + positive_stride * position)
        ix_from = int(ix_center_shift - kernel / 2)
        ix_to = int(ix_from + kernel)

        samples_buffer[buffer_count, :-8] = flux_normalized[ix_from:ix_to]
        samples_buffer[buffer_count, -1] = 1
        samples_buffer[buffer_count, -2] = col_density
        samples_buffer[buffer_count, -3] = central_wavelength
        samples_buffer[buffer_count, -4] = plate
        samples_buffer[buffer_count, -5] = mjd
        samples_buffer[buffer_count, -6] = fiber
        samples_buffer[buffer_count, -7] = ra
        samples_buffer[buffer_count, -8] = dec
        buffer_count += 1

    return samples_buffer[0:buffer_count, :]


def percent_nan_in_dla_range(flux, loglam, z_qso):
    lam, lam_rest, ix_dla_range = get_lam_data(loglam, z_qso, REST_RANGE)
    return float(np.sum(np.isnan(flux[ix_dla_range]))) / float(np.sum(ix_dla_range))


def prepare_density_regression_train_test(kernel=400, pos_sample_kernel_percent=0.2, n_samples=80,
                                          train_keys_csv="../data/dr9_train_set.csv",
                                          test_keys_csv="../data/dr9_test_set.csv",
                                          train_save_file="../data/densitydata_train.npy",
                                          test_save_file="../data/densitydata_test.npy"):
    dr9_train = np.genfromtxt(train_keys_csv, delimiter=',')
    dr9_test = np.genfromtxt(test_keys_csv, delimiter=',')

    data_train = np.zeros((2000000, kernel + 8), np.float32)
    data_test = np.zeros((2000000, kernel + 8), np.float32)
    loc_train = 0
    loc_test = 0

    # Training set
    for i in range(0, np.shape(dr9_train)[0]):
        try:
            data1, z_qso = read_fits_file(dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2])
            flux_norm = normalize(data1, z_qso)

            data_pos = scan_flux_about_central_wavelength(flux_norm, data1['loglam'], z_qso,
                                                          dr9_train[i, 3], dr9_train[i, 4], n_samples,
                                                          dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2], -1, -1,
                                                          kernel=kernel,
                                                          pos_sample_kernel_percent=pos_sample_kernel_percent)
            data_train[loc_train:loc_train + np.shape(data_pos)[0], :] = data_pos
            loc_train += np.shape(data_pos)[0]

            print(loc_train, np.shape(data_pos), np.shape(data_train))
        except Exception as e:
            print("Error ecountered on sample: ", dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2])
            print_exc()
            raise e

    # Test set
    for i in range(0, np.shape(dr9_test)[0]):
        try:
            data1, z_qso = read_fits_file(dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2])
            flux_norm = normalize(data1, z_qso)

            data_pos = scan_flux_about_central_wavelength(flux_norm, data1['loglam'], z_qso,
                                                          dr9_test[i, 3], dr9_test[i, 4], n_samples,
                                                          dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2], -1, -1,
                                                          kernel=kernel,
                                                          pos_sample_kernel_percent=pos_sample_kernel_percent)
            data_test[loc_test:loc_test + np.shape(data_pos)[0], :] = data_pos
            loc_test += np.shape(data_pos)[0]

            percent_nan = percent_nan_in_dla_range(flux_norm, data1['loglam'], z_qso)
            if percent_nan > 0.25:
                print("%0.3f,%d,%d,%d,%f,%f" %
                      (percent_nan, dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2], dr9_test[i, 3], dr9_test[i, 4]))

            print(loc_test, np.shape(data_pos), np.shape(data_test), 'nans', np.sum(np.isnan(data_pos[:-8])))
        except Exception as e:
            print("Error ecountered on sample: ", dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2])
            print_exc()
            raise e

    np.save(train_save_file, data_train[0:loc_train, :])
    np.save(test_save_file, data_test[0:loc_test, :])
    return DataSet(data_train[0:loc_train, :]), DataSet(data_test[0:loc_test, :])


# Get the flux from the raw data normalized, truncated, and padded for classification
def get_raw_data_for_classification(data1, z_qso, label=-1, central_wavelength=-1, col_density=-1,
                                    plate=-1, mjd=-1, fiber=-1, ra=-1, dec=-1):
    # Clean up the flux data
    flux_norm = normalize(data1, z_qso)

    lam = 10.0 ** data1['loglam']
    lam_rest = lam / (1 + z_qso)
    ix_rest_range = np.logical_and(lam_rest >= REST_RANGE[0], lam_rest <= REST_RANGE[1])

    # print("ix_rest_range in get_raw_data_for_classification: ", np.sum(ix_rest_range))
    npy = np.zeros((1, REST_RANGE[2] + 8), dtype=np.float32)
    npy[0, :REST_RANGE[2]] = np.pad(flux_norm[ix_rest_range],
                                    (max(0, REST_RANGE[2] - np.sum(ix_rest_range)), 0),
                                    'constant')
    npy[0, -1] = label
    npy[0, -2] = central_wavelength
    npy[0, -3] = col_density
    npy[0, -4] = plate
    npy[0, -5] = mjd
    npy[0, -6] = fiber
    npy[0, -7] = ra
    npy[0, -8] = dec

    return npy


def prepare_classification_training_set(train_csv_file="../data/classification_train.csv",
                                        test_dla_csv_file="../data/classification_test_dla.csv",
                                        test_non_dla_csv_file="../data/classification_test_non_dla.csv",
                                        save_train_file="../data/classification_train.npy",
                                        save_test_dla_file="../data/classification_test_dla.npy",
                                        save_test_non_dla_file="../data/classification_test_non_dla.npy"):
    for (csv_file, save_file) in zip([train_csv_file, test_dla_csv_file, test_non_dla_csv_file],
                                     [save_train_file, save_test_dla_file, save_test_non_dla_file]):
        csv = np.genfromtxt(csv_file, delimiter=',')
        # -3 labels mean sub-dla, count them as non-dla (0) in this pipeline
        csv[csv[:, 4] == -3, 4] = 0
        # Remove samples who's label is not 0 or 1 from the set
        csv = csv[csv[:, 4] >= 0, :]

        npy = np.zeros((np.shape(csv)[0], REST_RANGE[2] + 8), dtype=np.float32)

        for i in range(0, np.shape(csv)[0]):
            if i % 200 == 0 or i == np.shape(csv)[0] - 1:
                print("Processed %d samples in file %s" % (i, csv_file))
            (data1, z_qso) = read_fits_file(csv[i, 0], csv[i, 1], csv[i, 2])
            flux_for_classification = get_raw_data_for_classification(data1, z_qso, label=csv[i, 4],
                                                                      plate=csv[i, 0], mjd=csv[i, 1], fiber=csv[i, 2])

            # Pad the array to the left with zero's if it is short (REST_RANGE[2] in length), e.g. the sightline
            # doesn't reach 920A in the rest frame
            npy[i, :] = flux_for_classification

        print("Saving file %s" % save_file)
        np.save(save_file, npy)


# Read fits files and prepare data into numpy files with train/test splits
def prepare_localization_training_set(kernel=400, stride=3, pos_sample_kernel_percent=0.3,
                                      train_keys_csv="../data/dr9_train_set.csv",
                                      test_keys_csv="../data/dr9_test_set.csv",
                                      train_save_file="../data/localize_train.npy",
                                      test_save_file="../data/localize_test.npy"):
    dr9_train = np.genfromtxt(train_keys_csv, delimiter=',')
    dr9_test = np.genfromtxt(test_keys_csv, delimiter=',')

    data_train = np.zeros((4000000, kernel + 8), np.float32)
    data_test = np.zeros((4000000, kernel + 8), np.float32)
    loc_train = 0
    loc_test = 0

    # Training set
    for i in range(0, np.shape(dr9_train)[0]):
        try:
            data1, z_qso = read_fits_file(dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2])
            flux_norm = normalize(data1, z_qso)

            data_neg = scan_flux_sample(flux_norm, data1['loglam'], z_qso, dr9_train[i, 3], dr9_train[i, 4],
                                        dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2], -1, -1,
                                        exclude_positive_samples=True, kernel=kernel, stride=stride,
                                        pos_sample_kernel_percent=pos_sample_kernel_percent)
            data_train[loc_train:loc_train + np.shape(data_neg)[0], :] = data_neg
            loc_train += np.shape(data_neg)[0]

            data_pos = scan_flux_about_central_wavelength(flux_norm, data1['loglam'], z_qso,
                                                          dr9_train[i, 3], dr9_train[i, 4], np.shape(data_neg)[0],
                                                          dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2], -1, -1,
                                                          kernel=kernel,
                                                          pos_sample_kernel_percent=pos_sample_kernel_percent)
            data_train[loc_train:loc_train + np.shape(data_pos)[0], :] = data_pos
            loc_train += np.shape(data_pos)[0]

            print(loc_train, np.shape(data_neg), np.shape(data_pos), np.shape(data_train))
            # data_train = np.vstack((data_train,data_neg,data_pos))
        except Exception as e:
            print("Error ecountered on sample: ", dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2])
            print_exc()
            raise e

    # Test set
    for i in range(0, np.shape(dr9_test)[0]):
        try:
            data1, z_qso = read_fits_file(dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2])
            flux_norm = normalize(data1, z_qso)

            data_neg = scan_flux_sample(flux_norm, data1['loglam'], z_qso, dr9_test[i, 3], dr9_test[i, 4],
                                        dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2], -1, -1,
                                        exclude_positive_samples=True, kernel=400, stride=5,
                                        pos_sample_kernel_percent=0.3)
            data_test[loc_test:loc_test + np.shape(data_neg)[0], :] = data_neg
            loc_test += np.shape(data_neg)[0]

            data_pos = scan_flux_about_central_wavelength(flux_norm, data1['loglam'], z_qso,
                                                          dr9_test[i, 3], dr9_test[i, 4], np.shape(data_neg)[0],
                                                          dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2], -1, -1,
                                                          kernel=400, pos_sample_kernel_percent=0.3)
            data_test[loc_test:loc_test + np.shape(data_pos)[0], :] = data_pos
            loc_test += np.shape(data_pos)[0]

            print(loc_test, np.shape(data_neg), np.shape(data_pos), np.shape(data_test))
            # data_test = np.vstack((data_test,data_neg,data_pos))
        except Exception as e:
            print("Error ecountered on sample: ", dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2])
            print_exc()
            raise e

    np.save(train_save_file, data_train[0:loc_train, :])
    np.save(test_save_file, data_test[0:loc_test, :])
    return DataSet(data_train[0:loc_train, :]), DataSet(data_test[0:loc_test, :])


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
        #
        # Batch read fits file or from Igm library
        process_timer = timeit.default_timer()
        print "Reading up to %d fits files with %d cores" % (CHUNK_SIZE, multiprocessing.cpu_count() - 2)
        p = Pool(multiprocessing.cpu_count() - 2)
        if load_catalog == 'DR12_FITS':
            data1_zqso_tuple = p.map(multiprocess_read_fits_data, map(tuple, csv[i:i + CHUNK_SIZE, 0:3]))
        elif load_catalog == 'DR7_IGM':
            data1_zqso_tuple = p.map(multiprocess_read_igmspec, map(tuple, csv[i:i + CHUNK_SIZE, 0:2]))
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
        c2_buffer = np.zeros((REST_RANGE[2] * buff_size, kernel_size + 8), dtype=np.float32)
        r1_buffer = []
        c2_count = 0

        #
        # Batch pre-process data sets into proper numpy formats for C1, C2, and R1
        #
        gc.disable()  # Work around garbage collection bug: https://goo.gl/8YMYPH
        for ix, (data1, z_qso) in zip(range(buff_size), data1_zqso_tuple):
            print "Processing ", data1['plate'], data1['mjd'], data1['fiber']       # Debug find bad fits files
            c1_data = get_raw_data_for_classification(data1, z_qso,
                                                      plate=data1['plate'], mjd=data1['mjd'], fiber=data1['fiber'],
                                                      ra=data1['ra'], dec=data1['dec'])
            c2_data = scan_flux_sample(normalize(data1, z_qso), data1['loglam'], z_qso, -1, -1,
                                       data1['plate'], data1['mjd'], data1['fiber'], data1['ra'], data1['dec'],
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
        c1_pred, c1_conf, loc_pred, loc_conf, peaks_data, density_pred = \
            process_pipeline_for_batch(c1_buffer, c2_buffer, z_qso_buffer, loglam_buffer, r1_buffer,
                                       MODEL_CHECKPOINT_C1, MODEL_CHECKPOINT_C2, MODEL_CHECKPOINT_R1, kernel_size)

        #
        # Process output for each sightline
        #
        dla_counter = 0
        for ix in range(np.shape(c1_pred)[0]):
            # Store classification level data in results
            sightline = ({
                'plate': int(c1_buffer[ix, -4]), 'mjd': int(c1_buffer[ix, -5]), 'fiber': int(c1_buffer[ix, -6]),
                'ra':float(c1_buffer[ix, -7]), 'dec':float(c1_buffer[ix, -8]),
                'classification': "HAS_DLA" if c1_pred[ix] else "NO_DLA",
                'classification_confidence': int((abs(0.5 - c1_conf[ix]) + 0.5) * 100),
                'z_qso': float(z_qso_buffer[ix]),
                'num_dlas': len(peaks_data[ix][0]),
                'dlas': []
            })

            # Loop through peaks
            (peaks, peaks_uncentered, smoothed_sample, ixs_left, ixs_right) = peaks_data[ix]
            for peak in peaks:
                # lam, lam_rest, ix_dla_range = get_lam_data(data1['loglam'], z_qso_buffer[ix], REST_RANGE)
                lam, lam_rest, ix_dla_range = get_lam_data(loglam_buffer[ix,:], z_qso_buffer[ix], REST_RANGE)
                peak_lam_rest = lam_rest[ix_dla_range][peak]
                peak_lam_spectrum = peak_lam_rest * (1 + z_qso_buffer[ix])

                mean_col_density_prediction = np.mean(density_pred[:, dla_counter:dla_counter + 80])
                std_col_density_prediction = np.std(density_pred[:, dla_counter:dla_counter + 80])
                dla_counter += 80
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
        assert dla_counter == density_pred.shape[1]
        print "Processing PDFs"
        num_cores = multiprocessing.cpu_count() - 1
        # print c1_count
        split_count = int(math.ceil(float(np.shape(c1_buffer)[0]) / num_cores))
        split = range(split_count, np.shape(c1_buffer)[0], split_count)
        p = Pool(num_cores)

        density_pred_per_dla_split = 80 * np.cumsum([len(pd[0]) for pd in peaks_data])

        # sum every set of density predictions per dla, this # will vary by the # of dla's per prediction
        density_pred_per_core_split = [j[-1] for j in np.split(density_pred_per_dla_split, split)][:-1]

        z1 = np.split(c1_buffer, split)
        z2 = np.split(c2_buffer, split)
        z3 = np.split(loglam_buffer, split)
        z4 = np.split(z_qso_buffer, split)
        z5 = np.split(c1_pred, split)
        z6 = np.split(c1_conf, split)
        z7 = np.array_split(peaks_data, split)
        z8 = np.split(np.array(np.split(loc_conf, np.shape(c1_buffer)[0])), split)
        z9 = np.split(density_pred, density_pred_per_core_split, axis=1)
        z10= np.array_split(r1_buffer, split)
        z = zip(z1, z2, z3, z4, z5, z6, z7, z8, z9, z10)
        assert len(z1) == len(z2) == len(z3) == len(z4) == len(z5) == len(z6) == len(z7) == len(z8) == len(z9) == len(z10), \
            "Length of PDF parameters don't match: %d %d %d %d %d %d %d %d %d %d" % \
            (len(z1), len(z2), len(z3), len(z4), len(z5), len(z6), len(z7), len(z8), len(z9), len(z10))
        assert len(z) <= num_cores
        p.map(generate_pdfs, z)
        p.close()
        p.join()

        print "Processed %d sightlines for reporting on %d cores in %0.2fs" % \
              (np.shape(c1_pred)[0], num_cores, timeit.default_timer() - report_timer)

        runtime = timeit.default_timer() - process_timer
        print "Processed %d of %d in %0.0fs - %0.2fs per sample" % \
              (i + CHUNK_SIZE, np.shape(csv)[0], runtime, runtime / CHUNK_SIZE)

    # Write JSON string
    with open("../tmp/pipeline_predictions.json", 'w') as outfile:
        json.dump(sightline_results, outfile, indent=4)


# Generates a set of PDF visuals for each sightline and predictions
def generate_pdfs((c1_buffer, c2_buffer, loglam_buffer, z_qso_buffer, c1_pred, c1_conf,
                  peaks_data, loc_conf, density_pred, r1_buffer)):
    PLOT_LEFT_BUFFER = 50       # The number of pixels to plot left of the predicted sightline
    dlas_counter = 0

    for i in range(np.shape(c1_buffer)[0]):
        filename = "../tmp/visuals/dla-spec-%04d-%05d-%04d.pdf" % (c1_buffer[i, -4], c1_buffer[i, -5], c1_buffer[i, -6])
        pp = PdfPages(filename)

        (peaks, peaks_uncentered, smoothed_sample, ixs_left, ixs_right) = peaks_data[i]
        lam, lam_rest, ix_dla_range = get_lam_data(loglam_buffer[i, :], z_qso_buffer[i], REST_RANGE)
        full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(r1_buffer[i]['loglam'], z_qso_buffer[i], REST_RANGE)
        # x = lam_rest
        xlim = [REST_RANGE[0]-PLOT_LEFT_BUFFER, lam_rest[-1]]
        # y = c1_buffer[i, :-8]
        y = r1_buffer[i]['flux']
        y_plot_range = np.mean(y[y > 0]) * 3.0
        ylim = [-2, y_plot_range]

        n_dlas = len(peaks_data[i][0])
        n_plots = n_dlas + 3
        axtxt = 0
        axsl = 1
        axloc = 2

        # Plot DLA range
        fig, ax = plt.subplots(n_plots, figsize=(20, (3.75 * n_plots) + (0.1 * n_dlas) + 0.15), sharex=False)

        ax[axsl].set_xlabel("Rest frame sightline in region of interest for DLAs with z_qso = [%0.4f]" % z_qso_buffer[i])
        ax[axsl].set_ylabel("Flux")
        ax[axsl].set_ylim(ylim)
        ax[axsl].set_xlim(xlim)
        ax[axsl].plot(full_lam_rest, r1_buffer[i]['flux'], '-k')

        # Plot z_qso line over sightline
        ax[axsl].plot((1216, 1216), (ylim[0], ylim[1]), 'k--')

        # Plot localization
        ax[axloc].set_xlabel("DLA Localization confidence & localization prediction(s)")
        ax[axloc].set_ylabel("Confidence")
        ax[axloc].plot(lam_rest, loc_conf[i, :], color='deepskyblue')
        ax[axloc].set_ylim([0, 1])
        ax[axloc].set_xlim(xlim)

        # Classification results
        textresult = "Classified %d-%d-%d (%0.2f ra / %0.2f dec) as %s with confidence of %d%% (50%%=guess, 100%%=confident)\n" \
                     % (c1_buffer[i, -4], c1_buffer[i, -5], c1_buffer[i, -6], c1_buffer[i, -7], c1_buffer[i, -8],
                        "HAS_DLA" if c1_pred[i] else "NO_DLA", int((abs(0.5 - c1_conf[i]) + 0.5) * 100))

        for pltix, peak, peak_uncentered, ix_left, ix_right in \
                zip(range(axloc + 1, n_plots), peaks, peaks_uncentered, ixs_left, ixs_right):

            # Plot smoothed line
            ax[axloc].plot(lam_rest, smoothed_sample, color='blue', alpha=0.9)

            # Plot peak '+' markers
            ctrl_pts_height = (smoothed_sample[ix_left] + smoothed_sample[ix_right]) / 2
            ax[axloc].plot(lam_rest[peak_uncentered], loc_conf[i, peak_uncentered], '+', mew=3, ms=7, color='red', alpha=1)
            ax[axloc].plot(lam_rest[peak], smoothed_sample[peak], '+', mew=7, ms=15, color='blue', alpha=0.8)
            ax[axloc].plot(lam_rest[ix_left], ctrl_pts_height, '+', mew=3, ms=7, color='orange', alpha=1)
            ax[axloc].plot(lam_rest[ix_right], ctrl_pts_height, '+', mew=3, ms=7, color='orange', alpha=1)

            # Sightline plot transparent marker boxes
            ax[axsl].fill_between(lam_rest[ix_dla_range][peak - 10:peak + 10], y_plot_range, -2, color='gray', lw=0, alpha=0.1)
            ax[axsl].fill_between(lam_rest[ix_dla_range][peak - 30:peak + 30], y_plot_range, -2, color='gray', lw=0, alpha=0.1)
            ax[axsl].fill_between(lam_rest[ix_dla_range][peak - 50:peak + 50], y_plot_range, -2, color='gray', lw=0, alpha=0.1)
            ax[axsl].fill_between(lam_rest[ix_dla_range][peak - 70:peak + 70], y_plot_range, -2, color='gray', lw=0, alpha=0.1)

            # Plot column density measures with bar plots
            density_pred_per_this_dla = density_pred[:, dlas_counter * 80:dlas_counter * 80 + 80]
            dlas_counter += 1
            mean_col_density_prediction = np.mean(density_pred_per_this_dla)
            ax[pltix].bar(np.arange(0, np.shape(density_pred_per_this_dla)[1]), density_pred_per_this_dla[0, :], 0.25)
            ax[pltix].set_xlabel("Individual Column Density estimates for peak @ %0.0fA, +/- 0.3 of mean. " %
                                 (lam_rest[peak]) +
                                 "Mean: %0.3f - Median: %0.3f - Stddev: %0.3f" %
                                 (mean_col_density_prediction, np.median(density_pred_per_this_dla),
                                  np.std(density_pred_per_this_dla)))
            ax[pltix].set_ylim([mean_col_density_prediction - 0.3, mean_col_density_prediction + 0.3])
            ax[pltix].plot(np.arange(0, np.shape(density_pred_per_this_dla)[1]),
                           np.ones((np.shape(density_pred_per_this_dla)[1],), np.float32) * mean_col_density_prediction)

            ax[axloc].legend(['DLA pred', 'Smoothed pred', 'Original peak', 'Recentered peak', 'Centering points'],
                             bbox_to_anchor=(1.0, 1.1))

            # Add DLA to test result
            textresult += \
                " > DLA central wavelength at: %0.0fA rest / %0.0fA spectrum w/ confidence %0.2f, has Column Density: %0.3f\n" \
                % (lam_rest[peak], lam_rest[peak] * (1 + z_qso_buffer[i]), smoothed_sample[peak], mean_col_density_prediction)

        # Display text
        ax[axtxt].text(0, 0, textresult, family='monospace', fontsize='xx-large')
        ax[axtxt].get_xaxis().set_visible(False)
        ax[axtxt].get_yaxis().set_visible(False)
        ax[axtxt].set_frame_on(False)

        plt.tight_layout()
        pp.savefig(figure=fig)
        pp.close()
        plt.close(fig)

    assert dlas_counter * 80 == np.shape(density_pred)[1], \
        "dlas_counter, shape density_pred[1]: %d %d" % (
        dlas_counter, np.shape(density_pred)[1])  # Sanity check for bugs


def process_pipeline_for_batch(c1_buffer, c2_buffer, z_qso_buffer, loglam_buffer, r1_buffer,
                               MODEL_CHECKPOINT_C1, MODEL_CHECKPOINT_C2, MODEL_CHECKPOINT_R1, kernel_size):
    # Classification & localization models - Load model hyperparameter file & Generate predictions
    with open(MODEL_CHECKPOINT_C1 + "_hyperparams.json", 'r') as fp:
        hyperparameters_c1 = json.load(fp)
    with open(MODEL_CHECKPOINT_C2 + "_hyperparams.json", 'r') as fp:
        hyperparameters_c2 = json.load(fp)
    with open(MODEL_CHECKPOINT_R1 + "_hyperparams.json", 'r') as fp:
        hyperparameters_r1 = json.load(fp)

    #
    # Model Predictions performed in batch
    #
    print "C1 predictions begin"
    c1_pred, c1_conf = predictions_ann_c1(hyperparameters_c1, c1_buffer[:, :-8], c1_buffer[:, -1], MODEL_CHECKPOINT_C1)
    print "C2 predictions begin"
    loc_pred, loc_conf = predictions_ann_c2(hyperparameters_c2, c2_buffer[:, :-8], c2_buffer[:, -1],
                                            MODEL_CHECKPOINT_C2)
    print "Predictions to central wavelength begin"
    predtocent_timer = timeit.default_timer()
    peaks_data = predictions_to_central_wavelength(loc_conf, np.shape(c1_pred)[0], 50, 300)
    print "Complete predictions to central wavelength in %0.1f" % (timeit.default_timer() - predtocent_timer)

    n_density_est = 80
    dla_count = np.sum([len(peak_data[0]) for peak_data in peaks_data])
    density_data = np.zeros((dla_count * n_density_est, kernel_size + 8), dtype=np.float32)
    r1_count = 0

    # Loop through peaks generating full set of density estimate data samples
    print "Processing peaks data begin"
    for ix, (peaks, peaks_uncentered, smoothed_sample, ixs_left, ixs_right) in \
            zip(range(np.shape(c1_pred)[0]), peaks_data):
        for peak in peaks:
            z_qso = z_qso_buffer[ix]
            lam, lam_rest, ix_dla_range = get_lam_data(loglam_buffer[ix, :], z_qso, REST_RANGE)
            peak_lam_rest = lam_rest[ix_dla_range][peak]

            density_data[r1_count:r1_count + n_density_est, :] = \
                scan_flux_about_central_wavelength(r1_buffer[ix]['flux'], r1_buffer[ix]['loglam'], z_qso_buffer[ix],
                                                   peak_lam_rest * (1 + z_qso), 0, n_density_est, 0, 0, 0, -1, -1,
                                                   kernel_size, 0.2)
            r1_count += n_density_est

    # Column density model predictions
    print "R1 predictions begin"
    density_pred = predictions_ann_r1(hyperparameters_r1, density_data[:, :-8], c1_buffer[-1], MODEL_CHECKPOINT_R1)

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
    return c1_pred, c1_conf, loc_pred, loc_conf, peaks_data, density_pred

listen()    # Adds a user signal listener for debugging purposes.