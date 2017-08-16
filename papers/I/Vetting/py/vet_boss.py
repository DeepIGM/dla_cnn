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
    # Load BOSS ML
    _, dr12_abs = load_ml_dr12()
    # Cut on DLA
    dlas = dr12_abs['NHI'] >= 20.3
    dr12_dla = dr12_abs[dlas]

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
    missing_dr12 = dr12_dla[~matched][high_NHI_ML]

    # Not a z match?
    missing_dr12['dz16'] = 99.
    missing_dr12_coord = SkyCoord(ra=missing_dr12['RA'], dec=missing_dr12['DEC'], unit='deg')
    for kk,row in enumerate(missing_dr12):
        d2d = missing_dr12_coord[kk].separation(g16_coord)
        mt = d2d < 1*u.arcsec
        if np.sum(mt) > 0:
            zmin = np.abs(row['zabs']-g16_dlas['z_DLA'][mt])
            missing_dr12['dz16'][kk] = zmin
    # missing_dr12[['Plate','Fiber','zem','zabs','NHI','dz16']].more()
    pdb.set_trace()

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
