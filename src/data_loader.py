import math
import numpy as np
import os
import urllib
from traceback import print_exc

from astropy.io import fits


# DLAs from the DR9 catalog range from 920 to 1214, adding 120 on the right for errors in ly-a
# the last number is the number of pixels in SDSS sightlines that span the range
REST_RANGE = [920, 1334, 1614]


class DataSet:
    def __init__(self, raw_data):
        """Construct a DataSet"""
        self._fluxes = raw_data[:, :-6]
        self._labels = raw_data[:, -1]
        self._col_density = raw_data[:, -2]
        self._central_wavelength = raw_data[:, -3]
        self._plate = raw_data[:, -4]
        self._mjd = raw_data[:, -5]
        self._fiber = raw_data[:, -6]

        self._fluxes[np.isnan(self._fluxes)] = 0  # TODO change this to interpolate

        self._samples_consumed = 0
        self._ix_permutation = np.random.permutation(np.shape(self._labels)[0])

    @property
    def fluxes(self):
        return self._fluxes

    @property
    def labels(self):
        return self._labels

    @property
    def col_density(self):
        return self._col_density

    @property
    def plate(self):
        return self._plate

    @property
    def mjd(self):
        return self._mjd

    @property
    def fiber(self):
        return self._fiber

    def next_batch(self, batch_size):
        batch_ix = self._ix_permutation[0:batch_size]
        self._ix_permutation = np.roll(self._ix_permutation, batch_size)

        # keep track of how many samples have been consumed and reshuffle after an epoch has elapsed
        self._samples_consumed += batch_size
        if self._samples_consumed > np.shape(self._labels)[0]:
            self._ix_permutation = np.random.permutation(np.shape(self._labels)[0])
            self._samples_consumed = 0

        return self._fluxes[batch_ix, :], self._labels[batch_ix], self._col_density[batch_ix]


def normalize(data1, z_qso, divide_median=False):
    # flux
    flux = data1['flux']
    loglam = data1['loglam']
    lam = 10. ** loglam
    lam_rest = lam / (1. + z_qso)

    # npoints = np.shape(flux)[0]

    # pixel mask
    or_mask = data1['or_mask']
    bad_index = or_mask > 0
    flux[bad_index] = np.nan

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
    # data3 = fits_file[3].data
    z_qso = fits_file[3].data['LINEZ'][0]
    raw_data = {}

    # Pad loglam and flux_normalized to sufficiently below 920A rest that we don't have issues falling off the left
    (loglam_padded, flux_padded, or_mask_padded) = pad_loglam_flux(data1['loglam'], data1['flux'], data1['or_mask'], z_qso, 800)
    raw_data['flux'] = flux_padded
    raw_data['loglam'] = loglam_padded
    raw_data['or_mask'] = or_mask_padded

    return raw_data, z_qso


def pad_loglam_flux(loglam, flux, or_mask, z_qso, kernel):
    # Pad loglam and flux_normalized with kernel/2 zeros so scanning can start at 920A or higher
    pad_loglam_upper = round(math.log10((10**loglam[0])/(1+z_qso)),4)
    pad_loglam_lower = (math.floor(math.log10(REST_RANGE[0]) * 10000)-kernel/2) / 10000
    pad_loglam = np.linspace(pad_loglam_lower, pad_loglam_upper, max(0,(pad_loglam_upper-pad_loglam_lower+0.0001)*10000), dtype=np.float32)
    flux_padded = np.hstack((pad_loglam*0, flux))
    or_mask_padded = np.hstack((pad_loglam*0, or_mask))
    loglam_padded = np.hstack((pad_loglam, loglam))
    return loglam_padded, flux_padded, or_mask_padded


def scan_flux_sample(flux_normalized, loglam, z_qso, central_wavelength, col_density, mjd, plate, fiber,
                     exclude_positive_samples=False, kernel=400, stride=5, pos_sample_kernel_percent=0.3):
    # Split from rest frame 920A to 1214A (the range of DLAs in DR9 catalog)
    # pos_sample_kernel_percent is the percent of the kernel where positive samples can be found
    # e.g. the central wavelength is within this percentage of the center of the kernel window

    # Pre allocate space for generating samples
    samples_buffer = np.zeros((10000, kernel + 6), dtype=np.float32)
    buffer_count = 0

    # # Pad loglam and flux_normalized with kernel/2 zeros so scanning can start at 920A or higher
    # print('before', np.shape(loglam), np.shape(flux_normalized))
    # (loglam, flux_normalized) = pad_loglam_flux(loglam, flux_normalized, z_qso, kernel)
    # print('after', np.shape(loglam), np.shape(flux_normalized))

    lam = 10.0 ** loglam
    lam_rest = lam / (1. + z_qso)
    ix_dla_range = np.logical_and(lam_rest >= REST_RANGE[0], lam_rest <= REST_RANGE[1])
    ix_from = np.nonzero(ix_dla_range)[0][0]
    ix_to = np.shape(lam_rest)[0] - np.nonzero(np.flipud(ix_dla_range))[0][0]
    ix_central = np.nonzero(lam >= central_wavelength)[0][0]

    assert (ix_to > ix_central)

    # Scan across the data set generating negative samples
    # (skip positive samples where lam is near the central wavelength)
    for position in range(ix_from, ix_to, stride):
        if abs(position - ix_central) > kernel * pos_sample_kernel_percent:
            # Add a negative sample (not within pos_sample_kernel_percent of the central_wavelength)
            samples_buffer[buffer_count, :-6] = flux_normalized[position - kernel / 2:position - kernel / 2 + kernel]
            samples_buffer[buffer_count, -1] = 0
            samples_buffer[buffer_count, -2] = 0
            samples_buffer[buffer_count, -3] = 0
            samples_buffer[buffer_count, -4] = mjd
            samples_buffer[buffer_count, -5] = plate
            samples_buffer[buffer_count, -6] = fiber
            buffer_count += 1
        elif not exclude_positive_samples:
            # Add a positive sample (is within pos_sample_kernel_percent of the central_wavelength)
            samples_buffer[buffer_count, :-6] = flux_normalized[position - kernel / 2:position - kernel / 2 + kernel]
            samples_buffer[buffer_count, -1] = 1
            samples_buffer[buffer_count, -2] = col_density
            samples_buffer[buffer_count, -3] = central_wavelength
            samples_buffer[buffer_count, -4] = mjd
            samples_buffer[buffer_count, -5] = plate
            samples_buffer[buffer_count, -6] = fiber
            buffer_count += 1

    return samples_buffer[0:buffer_count, :]


def scan_flux_about_central_wavelength(flux_normalized, loglam, z_qso, central_wavelength, col_density,
                                       num_samples, mjd, plate, fiber, kernel=400, pos_sample_kernel_percent=0.3):
    samples_buffer = np.zeros((10000, kernel + 6), dtype=np.float32)
    buffer_count = 0

    # # Pad loglam and flux_normalized with kernel/2 zeros so scanning can start at 920A or higher
    # (loglam, flux_normalized) = pad_loglam_flux(loglam, flux_normalized, z_qso, kernel)

    lam = 10.0 ** loglam
    ix_central = np.nonzero(lam >= central_wavelength)[0][0]

    # Generate positive samples shifted left and right around the central wavelength
    positive_position_from = ix_central - kernel * pos_sample_kernel_percent / 2
    positive_position_to = positive_position_from + kernel * pos_sample_kernel_percent
    positive_stride = (positive_position_to - positive_position_from) / num_samples

    # TODO part of the issue with reading from left side
    # Adjust from index if the kernel would fall off the left side of the graph
    # if positive_position_from - kernel / 2 < 0:
    #     positive_position_from = positive_position_from + abs(kernel / 2 - positive_position_from)

    for position in range(0, num_samples):
        ix_center_shift = round(positive_position_from + positive_stride * position)
        ix_from = int(ix_center_shift - kernel / 2)
        ix_to = int(ix_from + kernel)

        samples_buffer[buffer_count, :-6] = flux_normalized[ix_from:ix_to]
        samples_buffer[buffer_count, -1] = 1
        samples_buffer[buffer_count, -2] = col_density
        samples_buffer[buffer_count, -3] = central_wavelength
        samples_buffer[buffer_count, -4] = mjd
        samples_buffer[buffer_count, -5] = plate
        samples_buffer[buffer_count, -6] = fiber
        buffer_count += 1

    return samples_buffer[0:buffer_count, :]


def percent_nan_in_dla_range(flux, loglam, z_qso):
    lam = 10.0 ** loglam
    lam_rest = lam / (1.0 + z_qso)
    ix_dla_range = np.logical_and(lam_rest >= REST_RANGE[0], lam_rest <= REST_RANGE[1])
    return float(np.sum(np.isnan(flux[ix_dla_range]))) / float(np.sum(ix_dla_range))


def prepare_density_regression_train_test(kernel=400, pos_sample_kernel_percent=0.2, n_samples=80,
                                          train_keys_csv="../data/dr9_train_set.csv",
                                          test_keys_csv="../data/dr9_test_set.csv",
                                          train_save_file="../data/densitydata_train.npy",
                                          test_save_file="../data/densitydata_test.npy"):
    dr9_train = np.genfromtxt(train_keys_csv, delimiter=',')
    dr9_test = np.genfromtxt(test_keys_csv, delimiter=',')

    data_train = np.zeros((2000000, kernel + 6), np.float32)
    data_test = np.zeros((2000000, kernel + 6), np.float32)
    loc_train = 0
    loc_test = 0

    # Training set
    for i in range(0, np.shape(dr9_train)[0]):
        try:
            data1, z_qso = read_fits_file(dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2])
            flux_norm = normalize(data1, z_qso)

            data_pos = scan_flux_about_central_wavelength(flux_norm, data1['loglam'], z_qso,
                                                          dr9_train[i, 3], dr9_train[i, 4], n_samples,
                                                          dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2],
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
                                                          dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2],
                                                          kernel=kernel,
                                                          pos_sample_kernel_percent=pos_sample_kernel_percent)
            data_test[loc_test:loc_test + np.shape(data_pos)[0], :] = data_pos
            loc_test += np.shape(data_pos)[0]

            percent_nan = percent_nan_in_dla_range(flux_norm, data1['loglam'], z_qso)
            if percent_nan > 0.25:
                print("%0.3f,%d,%d,%d,%f,%f" %
                      (percent_nan, dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2], dr9_test[i, 3], dr9_test[i, 4]))

            print(loc_test, np.shape(data_pos), np.shape(data_test), 'nans', np.sum(np.isnan(data_pos[:-6])))
        except Exception as e:
            print("Error ecountered on sample: ", dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2])
            print_exc()
            raise e

    np.save(train_save_file, data_train[0:loc_train, :])
    np.save(test_save_file, data_test[0:loc_test, :])
    return DataSet(data_train[0:loc_train, :]), DataSet(data_test[0:loc_test, :])


# # Localization predictions set, this scans the full DLA range creating creating a full range of samples to predict on
# def prepare_test_prediction_dataset(kernel=400, stride=1, pos_sample_kernel_percent=0.3, num_samples=10,
#                                       train_keys_csv="../data/dr9_train_set.csv",
#                                       test_keys_csv="../data/dr9_test_set.csv",
#                                       train_save_file="../data/densitydata_train.npy",
#                                       test_save_file="../data/densitydata_test.npy"):
#     dr9_test = np.genfromtxt(DR9_TEST_FNAME, delimiter=',')
#
#     data_buff = np.zeros((2000000, kernel + 6), np.float32)
#     loc = 0
#
#     for i in range(0,num_samples):
#         try:
#             data1, z_qso = read_fits_file(dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2])
#             flux_norm = normalize(data1, z_qso)
#
#             data = scan_flux_sample(flux_norm, data1['loglam'], z_qso, dr9_test[i, 3], dr9_test[i, 4],
#                                     dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2],
#                                     exclude_positive_samples=False, kernel=400, stride=stride,
#                                     pos_sample_kernel_percent=pos_sample_kernel_percent)
#             data_buff[loc:loc + np.shape(data)[0], :] = data
#             loc += np.shape(data)[0]
#
#             print(loc, np.shape(data_buff), 'ID', dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2])
#         except Exception as e:
#             print("Error ecountered on sample: ", dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2])
#             print_exc()
#             raise e
#
#     np.save("preddata_test.npy", data_buff[0:loc, :])
#     return DataSet(data[0:loc, :])


# Get the flux from the raw data normalized, truncated, and padded for classification
def get_raw_data_for_classification(data1, z_qso, label=-1, central_wavelength=-1, col_density=-1,
                                    plate=-1, mjd=-1, fiber=-1):
    # Clean up the flux data
    flux_norm = normalize(data1, z_qso)

    lam = 10.0 ** data1['loglam']
    lam_rest = lam / (1 + z_qso)
    ix_rest_range = np.logical_and(lam_rest >= REST_RANGE[0], lam_rest <= REST_RANGE[1])

    npy = np.zeros((1, REST_RANGE[2] + 6), dtype=np.float32)
    npy[0, :REST_RANGE[2]] = np.pad(flux_norm[ix_rest_range],
                                    (max(0, REST_RANGE[2] - np.sum(ix_rest_range)), 0),
                                    'constant')
    npy[0, -1] = label
    npy[0, -2] = central_wavelength
    npy[0, -3] = col_density
    npy[0, -4] = plate
    npy[0, -5] = mjd
    npy[0, -6] = fiber

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
        csv = csv[csv[:, 4] >= 0,:]

        npy = np.zeros((np.shape(csv)[0], REST_RANGE[2] + 6), dtype=np.float32)

        for i in range(0, np.shape(csv)[0]):
            if i % 200 == 0 or i == np.shape(csv)[0]-1:
                print("Processed %d samples in file %s" % (i, csv_file))
            (data1, z_qso) = read_fits_file(csv[i, 0], csv[i, 1], csv[i, 2])
            flux_for_classification = get_raw_data_for_classification(data1, z_qso)

            # Pad the array to the left with zero's if it is short (REST_RANGE[2] in length), e.g. the sightline
            # doesn't reach 920A in the rest frame
            npy[i, :] = flux_for_classification

        print("Saving file %s" % save_file)
        np.save(save_file, npy)


# Read fits files and prepare data into numpy files with train/test splits
def prepare_localization_training_set(kernel=400, stride=5, pos_sample_kernel_percent=0.3,
                                      train_keys_csv="../data/dr9_train_set.csv",
                                      test_keys_csv="../data/dr9_test_set.csv",
                                      train_save_file="../data/localize_train.npy",
                                      test_save_file="../data/localize_test.npy"):
    dr9_train = np.genfromtxt(train_keys_csv, delimiter=',')
    dr9_test = np.genfromtxt(test_keys_csv, delimiter=',')

    data_train = np.zeros((2000000, kernel + 6), np.float32)
    data_test = np.zeros((2000000, kernel + 6), np.float32)
    loc_train = 0
    loc_test = 0

    # Training set
    for i in range(0, np.shape(dr9_train)[0]):
        try:
            data1, z_qso = read_fits_file(dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2])
            flux_norm = normalize(data1, z_qso)

            data_neg = scan_flux_sample(flux_norm, data1['loglam'], z_qso, dr9_train[i, 3], dr9_train[i, 4],
                                        dr9_train[i, 0], dr9_train[i, 1], dr9_train[i, 2],
                                        exclude_positive_samples=True, kernel=kernel, stride=stride,
                                        pos_sample_kernel_percent=pos_sample_kernel_percent)
            data_train[loc_train:loc_train + np.shape(data_neg)[0], :] = data_neg
            loc_train += np.shape(data_neg)[0]

            data_pos = scan_flux_about_central_wavelength(flux_norm, data1['loglam'], z_qso,
                                                          dr9_train[i, 3], dr9_train[i, 4],
                                                          np.shape(data_neg)[0], dr9_train[i, 0], dr9_train[i, 1],
                                                          dr9_train[i, 2], kernel=kernel,
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
                                        dr9_test[i, 0], dr9_test[i, 1], dr9_test[i, 2],
                                        exclude_positive_samples=True, kernel=400, stride=5,
                                        pos_sample_kernel_percent=0.3)
            data_test[loc_test:loc_test + np.shape(data_neg)[0], :] = data_neg
            loc_test += np.shape(data_neg)[0]

            data_pos = scan_flux_about_central_wavelength(flux_norm, data1['loglam'], z_qso,
                                                          dr9_test[i, 3], dr9_test[i, 4],
                                                          np.shape(data_neg)[0], dr9_test[i, 0], dr9_test[i, 1],
                                                          dr9_test[i, 2], kernel=400, pos_sample_kernel_percent=0.3)
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


def load_data_sets(filename_train, filename_test):
    return DataSet(np.load(filename_train)), DataSet(np.load(filename_test))
