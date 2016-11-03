""" Module to vette results against Human catalogs
  SDSS-DR5 (JXP) and BOSS (Notredaeme)
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import pdb


from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u

from linetools import utils as ltu
from pyigm.surveys.dlasurvey import DLASurvey


def grab_sightlines(dlasurvey=None, flg_bal=1, s2n=5., DX=0.):
    """ Grab a set of sightlines without DLAs from a DLA survey

    Parameters
    ----------
    dlas : DLASurvey
      Usually SDSS or BOSS
    flg_bal : int, optional
      Maximum BAL flag (0=No signature, 1=Weak BAL, 2=BAL)
    s2n : float, optional
      Minimum S/N as defined in some manner
    DX : float, optional
      Restrict on DX

    Returns
    -------
    final : Table
      astropy Table of good sightlines
    sdict : dict
      dict describing the sightlines
    """
    # Init
    if dlasurvey is None:
        print("Using the DR5 sample for the sightlines")
        dlasurvey = DLASurvey.load_SDSS_DR5(sample='all')
    nsight = len(dlasurvey.sightlines)
    keep = np.array([True]*nsight)

    # Avoid DLAs
    dla_coord = dlasurvey.coord
    sl_coord = SkyCoord(ra=dlasurvey.sightlines['RA'], dec=dlasurvey.sightlines['DEC'])
    idx, d2d, d3d = match_coordinates_sky(sl_coord, dla_coord, nthneighbor=1)
    clear = d2d > 1*u.arcsec
    keep = keep & clear

    # BAL
    if flg_bal > 0:
        gd_bal = dlasurvey.sightlines['FLG_BAL'] <= flg_bal
        keep = keep & gd_bal

    # S/N
    if s2n > 0.:
        gd_s2n = dlasurvey.sightlines['S2N'] > s2n
        keep = keep & gd_s2n

    # Cut on DX
    if DX > 0.:
        gd_DX = dlasurvey.sightlines['DX'] > DX
        keep = keep & gd_DX

    # Assess
    final = dlasurvey.sightlines[keep]
    sdict = {}
    sdict['n'] = len(final)
    print("We have {:d} sightlines for analysis".format(sdict['n']))

    def qck_stats(idict, tbl, istr, key):
        idict[istr+'min'] = np.min(tbl[key])
        idict[istr+'max'] = np.max(tbl[key])
        idict[istr+'median'] = np.median(tbl[key])
    qck_stats(sdict, final, 'z', 'ZEM')
    qck_stats(sdict, final, 'i', 'MAG')

    print("Min z = {:g}, Median z = {:g}, Max z = {:g}".format(sdict['zmin'], sdict['zmedian'], sdict['zmax']))

    # Return
    return final, sdict

#def insert_a dla()


def main(flg_tst, sdss=None, ml_survey=None):

    # Sightlines
    if (flg_tst % 2**1) >= 2**0:
        if sdss is None:
            sdss = DLASurvey.load_SDSS_DR5(sample='all')
        slines, sdict = grab_sightlines(sdss)


# Test
if __name__ == '__main__':
    flg_tst = 0
    flg_tst += 2**0   # Grab sightlines
    #flg_tst += 2**1   # Vette

    main(flg_tst)
