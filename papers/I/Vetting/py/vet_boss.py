""" Module for vetting BOSS catalogs
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import sys
import io, json
import pdb
import numpy as np

from matplotlib import pyplot as plt

from pkg_resources import resource_filename

from astropy import units as u
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord, match_coordinates_sky

from linetools import utils as ltu
from pyigm.abssys.dla import DLASystem
from pyigm.abssys.lls import LLSSystem
from pyigm.surveys.llssurvey import LLSSurvey
from pyigm.surveys.dlasurvey import DLASurvey, dla_stat

from dla_cnn.io import load_ml_dr12, load_garnett16
from dla_cnn.catalogs import match_boss_catalogs


def highnhi_without_match():
    """ High confidence too (>0.9 in each)
    """
    # Load BOSS ML
    _, dr12_abs = load_ml_dr12()
    # Cut on DLA
    dlas = dr12_abs['NHI'] >= 20.3
    dr12_dla = dr12_abs[dlas]
    dr12_dla_coord = SkyCoord(ra=dr12_dla['RA'], dec=dr12_dla['DEC'], unit='deg')

    # Load Garnett
    g16_abs = load_garnett16()
    g16_dlas = g16_abs[g16_abs['log.NHI'] >= 20.3]
    g16_coord = SkyCoord(ra=g16_dlas['RAdeg'], dec=g16_dlas['DEdeg'], unit='deg')

    # Match
    dr12_to_g16 = match_boss_catalogs(dr12_dla, g16_dlas)
    matched = dr12_to_g16 >= 0
    g16_idx = dr12_to_g16[matched]

    # High conf, high NHI in ML but not in Garnett16
    #  And in Lya forest
    high_NHI_ML = (dr12_dla['conf'][~matched] > 0.9) & (dr12_dla['NHI'][~matched] > 21.) & (
        dr12_dla['flg_BAL'][~matched] == 0) & (dr12_dla['zabs'][~matched] > 2.) & (
        (1+dr12_dla['zabs'][~matched])*1215.67 > (1+dr12_dla['zem'][~matched])*1030.)
    print("There are {:d} high_NHI in ML but not in Garnett16 search path".format(np.sum(high_NHI_ML)))

    # Not a z match?
    missing_dr12 = dr12_dla[~matched][high_NHI_ML]
    missing_dr12['dz16'] = 99.
    missing_dr12_coord = SkyCoord(ra=missing_dr12['RA'], dec=missing_dr12['DEC'], unit='deg')
    for kk,row in enumerate(missing_dr12):  # This is slow
        d2d = missing_dr12_coord[kk].separation(g16_coord)
        mt = d2d < 1*u.arcsec
        if np.sum(mt) > 0:
            zmin = np.min(np.abs(row['zabs']-g16_dlas['z_DLA'][mt]))
            missing_dr12['dz16'][kk] = zmin
    # missing_dr12[['Plate','Fiber','zem','SNR','zabs','NHI','dz16']].more()

    # Match G16 to ML
    g16_to_ML = match_boss_catalogs(g16_dlas, dr12_dla, reverse=True)
    matched2 = g16_to_ML >= 0

    # Not matched, high quality DLA
    high_quality_DLA = (g16_dlas['pDLAD'][~matched2] > 0.9) & (g16_dlas['log.NHI'][~matched2] >= 20.3) & (
        g16_dlas['flg_BAL'][~matched2] == 0) & (g16_dlas['z_DLA'][~matched2] > 2.)
    print("There are {:d} high_quality DLA in G16 not in ML".format(np.sum(high_quality_DLA)))
    pdb.set_trace()

    veryhigh_NHI = (g16_dlas['pDLAD'][matched2] > 0.9) & (g16_dlas['log.NHI'][matched2] > 21.8) & (
        g16_dlas['flg_BAL'][matched2] == 0) & (g16_dlas['z_DLA'][matched2] > 2.)
    g16_matched = g16_dlas[matched2][veryhigh_NHI]
    g16_matched['ML_NHI'] = dr12_dla['NHI'][g16_to_ML[matched2][veryhigh_NHI]]
    g16_matched[['Plate','Fiber','z_QSO','SNR','z_DLA','log.NHI','ML_NHI']].write( 'G16_highNHI_matched.ascii', format='ascii.fixed_width', overwrite=True)

    # High conf, high NHI in G16 but not in ML
    high_NHI_G16 = (g16_dlas['pDLAD'][~matched2] > 0.9) & (g16_dlas['log.NHI'][~matched2] > 21.) & (
        g16_dlas['flg_BAL'][~matched2] == 0) & (g16_dlas['z_DLA'][~matched2] > 2.)
    print("There are {:d} high_NHI in G16 but not in ML".format(np.sum(high_NHI_G16)))
    missing_g16 = g16_dlas[~matched2][high_NHI_G16]
    missing_g16['dz12'] = 99.
    missing_g16['dNHI'] = 99.
    missing_g16_coord = SkyCoord(ra=missing_g16['RAdeg'], dec=missing_g16['DEdeg'], unit='deg')
    for kk,row in enumerate(missing_g16):  # This is very slow
        d2d = missing_g16_coord[kk].separation(dr12_dla_coord)
        mt = np.where(d2d < 1*u.arcsec)[0]
        if len(mt) > 0:
            idzmin = np.argmin(np.abs(row['z_DLA']-dr12_dla['zabs'][mt]))
            dzmin = row['z_DLA']-dr12_dla['zabs'][mt[idzmin]]
            dNHI = row['log.NHI']-dr12_dla['NHI'][mt[idzmin]]
            missing_g16['dz12'][kk] = dzmin
            missing_g16['dNHI'][kk] = dNHI
    missing_g16[['Plate','Fiber','z_QSO','SNR','z_DLA','log.NHI','dz12','dNHI']].write(
        'G16_highNHI_misses.ascii', format='ascii.fixed_width', overwrite=True)

def main(flg):

    if (flg & 2**0):  # Missing High NHI, high conf systems
        highnhi_without_match()

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1: #
        flg_vet = 0
        flg_vet += 2**0   # High NHI
        #flg_vet += 2**1   # Compare to N09
        #flg_vet += 2**2   # Compare to PW09
        #flg_vet += 2**3   # Compare to PW09
    else:
        flg_vet = int(sys.argv[1])

    main(flg_vet)
