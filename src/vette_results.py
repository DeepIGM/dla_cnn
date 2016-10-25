""" Module to vette results against Human catalogs
  SDSS-DR5 (JXP) and BOSS (Notredaeme)
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import pdb


import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u

from linetools import utils as ltu
from pyigm.surveys.llssurvey import LLSSurvey
from pyigm.surveys.dlasurvey import DLASurvey


def json_to_sdss_dlasurvey(json_file, sdss_survey):
    """ Convert JSON output file to a DLASurvey object
    Assumes SDSS bookkeeping for sightlines (i.e. PLATE, FIBER)

    Parameters
    ----------
    json_file : str
      Full path to the JSON results file
    sdss_survey : DLASurvey
      SDSS survey, usually human (e.g. JXP for DR5)

    Returns
    -------
    ml_survey : LLSSurvey
      Survey object for the LLS

    """
    print("Loading SDSS Survey from JSON file {:s}".format(json_file))
    # imports
    from pyigm.abssys.dla import DLASystem
    from pyigm.abssys.lls import LLSSystem
    # Fiber key
    for fkey in ['FIBER', 'FIBER_ID', 'FIB']:
        if fkey in sdss_survey.sightlines.keys():
            break
    # Read
    ml_results = ltu.loadjson(json_file)
    # Init
    idict = dict(plate=[], fiber=[], classification_confidence=[],
                 classification=[], ra=[], dec=[])
    ml_tbl = Table()
    ml_survey = LLSSurvey()
    systems = []
    in_ml = np.array([False]*len(sdss_survey.sightlines))
    # Loop
    for obj in ml_results:
        # Sightline
        for key in idict.keys():
            idict[key].append(obj[key])
        # DLAs
        for idla in obj['dlas']:
            """
            dla = DLASystem((sdss_survey.sightlines['RA'][mt[0]],
                             sdss_survey.sightlines['DEC'][mt[0]]),
                            idla['spectrum']/(1215.6701)-1., None,
                            idla['column_density'])
            """
            if idla['z_dla'] < 1.8:
                continue
            isys = LLSSystem((obj['ra'],obj['dec']),
                    idla['z_dla'], None, NHI=idla['column_density'], zem=obj['z_qso'])
            isys.confidence = idla['dla_confidence']
            # Save
            systems.append(isys)
    # Connect to sightlines
    ml_coord = SkyCoord(ra=idict['ra'], dec=idict['dec'], unit='deg')
    s_coord = SkyCoord(ra=sdss_survey.sightlines['RA'], dec=sdss_survey.sightlines['DEC'], unit='deg')
    idx, d2d, d3d = match_coordinates_sky(s_coord, ml_coord, nthneighbor=1)
    used = d2d < 1.*u.arcsec
    for iidx in idx[~used]:
        print("Sightline RA={:g}, DEC={:g} was not used".format(sdss_survey.sightlines['RA'][iidx],
                                                                sdss_survey.sightlines['DEC'][iidx]))
    # Finish
    ml_survey._abs_sys = systems
    ml_survey.sightlines = sdss_survey.sightlines[idx[used]]
    for key in idict.keys():
        ml_tbl[key] = idict[key]
    ml_survey.ml_tbl = ml_tbl
    # Return
    return ml_survey


def vette_dlasurvey(ml_survey, sdss_survey, fig_root='tmp', lyb_cut=True,
                    dz_toler=0.03):
    """
    Parameters
    ----------
    ml_survey : IGMSurvey
      Survey describing the Machine Learning results
    sdss_survey : DLASurvey
      SDSS survey, usually human (e.g. JXP for DR5)
    fig_root : str, optional
      Root string for figures generated
    lyb_cut : bool, optional
      Cut surveys at Lyb in QSO rest-frame.
      Recommended until LLS, Lyb and OVI is dealt with
    dz_toler : float, optional
      Tolerance for matching in redshift

    Returns
    -------
    false_neg : list
      List of systems that are false negatives from SDSS -> ML
    midx : list
      List of indices matching SDSS -> ML
    """
    from pyigm.surveys import dlasurvey as pyis_ds
    reload(pyis_ds)
    # Cut at Lyb
    if lyb_cut:
        for survey in [ml_survey, sdss_survey]:
            # Alter Z_START
            zlyb = (1+survey.sightlines['ZEM']).data*1026./1215.6701 - 1.
            survey.sightlines['Z_START'] = np.maximum(survey.sightlines['Z_START'], zlyb)
            # Mask
            mask = pyis_ds.dla_stat(survey, survey.sightlines)
            survey.mask = mask
        print("Done cutting on Lyb")

    # Setup coords
    ml_coords = ml_survey.coord
    ml_z = ml_survey.zabs
    #s_coords = sdss_survey.coord

    # Match from SDSS and record false negatives
    false_neg = []
    midx = []
    for igd in np.where(sdss_survey.mask)[0]:
        isys = sdss_survey._abs_sys[igd]
        # Match?
        gd_radec = np.where(isys.coord.separation(ml_coords) < 1*u.arcsec)[0]
        if len(gd_radec) == 0:
            false_neg.append(isys)
            midx.append(-1)
        else:
            gdz = np.abs(ml_z[gd_radec] - isys.zabs) < dz_toler
            if np.sum(gdz) > 0:
                iz = np.argmin(np.abs(ml_z[gd_radec] - isys.zabs))
                midx.append(gd_radec[iz])
            else:
                false_neg.append(isys)
                midx.append(-1)
    # Return
    return false_neg, np.array(midx)


def fig_dzdnhi(ml_survey, sdss_survey, midx, outfil='fig_dzdnhi.pdf'):
    """  Compare zabs and NHI between SDSS and ML

    Parameters
    ----------
    ml_survey : IGMSurvey
      Survey describing the Machine Learning results
      This should be masked according to the vetting
    sdss_survey : DLASurvey
      SDSS survey, usually human (e.g. JXP for DR5)
      This should be masked according to the vetting
    midx : list
      List of indices matching SDSS -> ML
    outfil : str, optional
      Input None to plot to screen

    Returns
    -------

    """
    # z, NHI
    z_sdss = sdss_survey.zabs
    z_ml = ml_survey.zabs
    NHI_sdss = sdss_survey.NHI
    NHI_ml = ml_survey.NHI
    # deltas
    dz = []
    dNHI = []
    for qq,idx in enumerate(midx):
        if idx < 0:
            continue
        # Match
        dz.append(z_sdss[qq]-z_ml[idx])
        dNHI.append(NHI_sdss[qq]-NHI_ml[idx])

    # Figure
    if outfil is not None:
        pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8, 5))
    plt.clf()
    gs = gridspec.GridSpec(1, 2)
    # dz
    ax = plt.subplot(gs[0])
    ax.hist(dz, color='green', bins=20)#, normed=True)#, bins=20 , zorder=1)
    #ax.text(0.05, 0.74, lbl3, transform=ax.transAxes, color=wcolor, size=csz, ha='left')
    ax.set_xlim(-0.03, 0.03)
    ax.set_xlabel(r'$\delta z$ [SDSS-ML]')
    # NHI
    ax = plt.subplot(gs[1])
    ax.hist(dNHI, color='blue', bins=20)#, normed=True)#, bins=20 , zorder=1)
    #ax.text(0.05, 0.74, lbl3, transform=ax.transAxes, color=wcolor, size=csz, ha='left')
    #ax.set_xlim(-0.03, 0.03)
    ax.set_xlabel(r'$\Delta \log N_{\rm HI}$ [SDSS-ML]')
    #
    # End
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    if outfil is not None:
        print('Writing {:s}'.format(outfil))
        pp.savefig()
        pp.close()
        plt.close()
    else:
        plt.show()


def fig_falseneg(ml_survey, sdss_survey, false_neg, outfil='fig_falseneg.pdf'):
    """   Figure on false negatives

    Parameters
    ----------
    ml_survey : IGMSurvey
      Survey describing the Machine Learning results
      This should be masked according to the vetting
    sdss_survey : DLASurvey
      SDSS survey, usually human (e.g. JXP for DR5)
      This should be masked according to the vetting
    midx : list
      List of indices matching SDSS -> ML
    false_neg : list
      List of false negatives
    outfil : str, optional
      Input None to plot to screen

    Returns
    -------

    """
    # Generate some lists
    zabs_false = [isys.zabs for isys in false_neg]
    zem_false = [isys.zem for isys in false_neg]
    NHI_false = [isys.NHI for isys in false_neg]

    # Figure
    if outfil is not None:
        pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8, 5))
    plt.clf()
    gs = gridspec.GridSpec(2, 2)
    # zabs
    ax = plt.subplot(gs[0])
    ax.hist(zabs_false, color='green', bins=20)#, normed=True)#, bins=20 , zorder=1)
    ax.set_xlabel(r'$z_{\rm abs}$')
    # zem
    ax = plt.subplot(gs[1])
    ax.hist(zem_false, color='red', bins=20)#, normed=True)#, bins=20 , zorder=1)
    ax.set_xlabel(r'$z_{\rm qso}$')
    # NHI
    ax = plt.subplot(gs[2])
    ax.hist(NHI_false, color='blue', bins=20)#, normed=True)#, bins=20 , zorder=1)
    ax.set_xlabel(r'$\log \, N_{\rm HI}$')
    # End
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    if outfil is not None:
        print('Writing {:s}'.format(outfil))
        pp.savefig()
        pp.close()
        plt.close()
    else:
        plt.show()



def main(flg_tst, sdss=None, ml_survey=None):

    # Load JSON for DR5
    if (flg_tst % 2**1) >= 2**0:
        if sdss is None:
            sdss = DLASurvey.load_SDSS_DR5()
        #ml_survey = json_to_sdss_dlasurvey('../results/dr5_v1_predictions.json', sdss)
        ml_survey = json_to_sdss_dlasurvey('../results/dr5_v2_results.json', sdss)

    # Vette
    if (flg_tst % 2**2) >= 2**1:
        if ml_survey is None:
            sdss = DLASurvey.load_SDSS_DR5()
            ml_survey = json_to_sdss_dlasurvey('../results/dr5_v2_results.json', sdss)
        vette_dlasurvey(ml_survey, sdss)

# Test
if __name__ == '__main__':
    flg_tst = 0
    #flg_tst += 2**0   # Load JSON for DR5
    flg_tst += 2**1   # Vette

    main(flg_tst)
