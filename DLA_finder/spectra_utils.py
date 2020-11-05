""" Methods for fussing with a spectrum """

import numpy as np
from dla_cnn.desi.defs import REST_RANGE

def get_lam_data(loglam, z_qso, REST_RANGE=REST_RANGE):
    """
    Generate wavelengths from the log10 wavelengths

    Parameters
    ----------
    loglam: np.ndarray
    z_qso: float
    REST_RANGE: list
        Lowest rest wavelength to search, highest rest wavelength,  number of pixels in the search

    Returns
    -------
    lam: np.ndarray
    lam_rest: np.ndarray
    ix_dla_range: np.ndarray
        Indices listing where to search for the DLA
    """
    #kernelrangepx = int(kernel/2)
    lam = 10.0 ** loglam
    lam_rest = lam / (1.0 + z_qso)
    ix_dla_range = np.logical_and(lam_rest >= REST_RANGE[0], lam_rest <= REST_RANGE[1])#&(lam>=3800) if low snr, only use lam>=3800

    return lam, lam_rest, ix_dla_range
