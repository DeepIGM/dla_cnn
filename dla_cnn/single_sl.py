""" Enable analysis of a single quasar sightline for DLAs"""
import numpy as np
from pkg_resources import resource_filename

from dla_cnn.data_model.Sightline import Sightline
from dla_cnn import data_loader

default_model = resource_filename('dla_cnn', "models/model_gensample_v7.1")


def gen_sightline(wave, flux, z_qso):
    """
    Only good for SDSS dispersion

    Args:
        wave:
        flux:
        z_qso:

    Returns:
        Sightline:

    """

    # Log and pad
    loglam = np.log10(np.array(wave))
    loglam_padded, flux_padded = data_loader.pad_loglam_flux(loglam, flux, z_qso)

    # Generate the sightline
    sightline = Sightline(id=0)
    sightline.flux = flux_padded
    sightline.loglam = loglam_padded
    sightline.z_qso = z_qso
    sightline.dlas = []

    # Validate
    data_loader.validate_sightline(sightline)

    return sightline

