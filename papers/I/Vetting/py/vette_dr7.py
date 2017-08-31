""" Module for vetting DR7 output
  Notredame07
  PW09 (DR5)
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

from dla_cnn.io import load_ml_dr7

dr7_file = resource_filename('dla_cnn', 'catalogs/sdss_dr7/predictions_SDSSDR7.json')

def profile():
    coord = SkyCoord(ra=12.231, dec=-12.2432, unit='deg')
    dla_list = []
    vlim = [-500., 500.]*u.km/u.s
    for ii in range(10000):
        if (ii % 100) == 0:
            print('tt: {:d}'.format(ii))
        isys = DLASystem(coord, 2., vlim, NHI=21.0, zem=3., name='dumb')
        dla_list.append(isys)
    return None


def chk_dr5_dla_to_ml(ml_dlasurvey=None, ml_llssurvey=None, dz_toler=0.015,
                      outfile='vette_dr5.json', write_again=True):
    # Load ML
    if (ml_dlasurvey is None) or (ml_llssurvey is None):
        ml_llssurvey, ml_dlasurvey = load_ml_dr7()
    # Load DR5
    dr5 = DLASurvey.load_SDSS_DR5()  # This is the statistical sample
    # Use coord to efficiently deal with sightlines
    ml_coord = SkyCoord(ra=ml_dlasurvey.sightlines['RA'], dec=ml_dlasurvey.sightlines['DEC'], unit='deg')
    dr5_coord = SkyCoord(ra=dr5.sightlines['RA'], dec=dr5.sightlines['DEC'], unit='deg')
    idx, d2d, d3d = match_coordinates_sky(dr5_coord, ml_coord, nthneighbor=1)
    in_ml = d2d < 2*u.arcsec
    print("{:d} of the DR5 sightlines were covered by ML out of {:d}".format(np.sum(in_ml), len(dr5.sightlines)))
    # 7477 sightlines out of 7482

    # Cut down
    dr5.sightlines = dr5.sightlines[in_ml]
    new_mask = dla_stat(dr5, dr5.sightlines) # 737 good DLAs
    dr5.mask = new_mask
    dr5_dla_coord = dr5.coord
    dr5_dla_zabs = dr5.zabs
    ndr5 = len(dr5_dla_coord)

    ml_dla_coord = ml_dlasurvey.coords
    ml_lls_coord = ml_llssurvey.coords

    # Loop on DR5 DLAs and save indices of the matches
    dr5_ml_idx = np.zeros(ndr5).astype(int) - 1
    for ii in range(ndr5):
        # Match to ML
        dla_mts = np.where(dr5_dla_coord[ii].separation(ml_dla_coord) < 2*u.arcsec)[0]
        nmt = len(dla_mts)
        if nmt == 0:  # No match
            # Check for LLS
            lls_mts = np.where(dr5_dla_coord[ii].separation(ml_lls_coord) < 2*u.arcsec)[0]
            nmt2 = len(lls_mts)
            if nmt2 == 0:  # No match
                pass
            else:
                zML = ml_llssurvey.zabs[lls_mts] # Redshifts of all DLAs on the sightline in ML
                zdiff = np.abs(dr5_dla_zabs[ii]-zML)
                if np.min(zdiff) < dz_toler:
                    dr5_ml_idx[ii] = -9  # SLLS match
        else:
            zML = ml_dlasurvey.zabs[dla_mts] # Redshifts of all DLAs on the sightline in ML
            zdiff = np.abs(dr5_dla_zabs[ii]-zML)
            if np.min(zdiff) < dz_toler:
                #print("Match on {:d}!".format(ii))
                # Match
                imin = np.argmin(zdiff)
                dr5_ml_idx[ii] = dla_mts[imin]
            else: # Check for LLS
                lls_mts = np.where(dr5_dla_coord[ii].separation(ml_lls_coord) < 2*u.arcsec)[0]
                nmt2 = len(lls_mts)
                if nmt2 == 0:  # No match
                    pass
                else:
                    zML = ml_llssurvey.zabs[lls_mts] # Redshifts of all DLAs on the sightline in ML
                    zdiff = np.abs(dr5_dla_zabs[ii]-zML)
                    if np.min(zdiff) < dz_toler:
                        dr5_ml_idx[ii] = -9  # SLLS match


    dr5_coord = SkyCoord(ra=dr5.sightlines['RA'], dec=dr5.sightlines['DEC'], unit='deg')

    # Write out misses
    misses = np.where(dr5_ml_idx == -1)[0]
    plates, fibers = [], []
    for miss in misses:
        imin = np.argmin(dr5_dla_coord[miss].separation(dr5_coord))
        plates.append(dr5.sightlines['PLATE'][imin])
        fibers.append(dr5.sightlines['FIB'][imin])
    mtbl = Table()
    mtbl['PLATE'] = plates
    mtbl['FIBER'] = fibers
    mtbl['NHI'] = dr5.NHI[misses]
    mtbl['zabs'] = dr5.zabs[misses]
    if write_again:
        mtbl.write('DR5_misses.ascii', format='ascii.fixed_width', overwrite=True)

    # Write out SLLS
    sllss = np.where(dr5_ml_idx == -9)[0]
    plates, fibers = [], []
    for slls in sllss:
        imin = np.argmin(dr5_dla_coord[slls].separation(dr5_coord))
        plates.append(dr5.sightlines['PLATE'][imin])
        fibers.append(dr5.sightlines['FIB'][imin])
    mtbl = Table()
    mtbl['PLATE'] = plates
    mtbl['FIBER'] = fibers
    mtbl['NHI'] = dr5.NHI[sllss]
    mtbl['zabs'] = dr5.zabs[sllss]
    if write_again:
        mtbl.write('DR5_SLLS.ascii', format='ascii.fixed_width', overwrite=True)

    # ML not matched by PW09?
    ml_dla_coords = ml_dlasurvey.coords
    idx2, d2d2, d3d = match_coordinates_sky(ml_dla_coords, dr5_dla_coord, nthneighbor=1)
    not_in_dr5 = d2d2 > 2*u.arcsec  # This doesn't match redshifts!
    might_be_in_dr5 = np.where(~not_in_dr5)[0]

    others_not_in = []  # this is some painful book-keeping
    for idx in might_be_in_dr5:  # Matching redshifts..
        imt = ml_dla_coord[idx].separation(dr5_dla_coord) < 2*u.arcsec
        # Match on dztoler
        if np.min(np.abs(ml_dlasurvey.zabs[idx]-dr5.zabs[imt])) > dz_toler:
            others_not_in.append(idx)

    # Save
    out_dict = {}
    out_dict['in_ml'] = in_ml
    out_dict['dr5_idx'] = dr5_ml_idx  # -1 are misses, -9 are SLLS
    out_dict['not_in_dr5'] = np.concatenate([np.where(not_in_dr5)[0], np.array(others_not_in)])
    ltu.savejson(outfile, ltu.jsonify(out_dict), overwrite=True)


def dr5_false_positives(ml_dlasurvey=None, ml_llssurvey=None):
    vette_file = 'vette_dr5.json'
    from pyigm.surveys.dlasurvey import DLASurvey
    from matplotlib import pyplot as plt
    # Load ML
    if (ml_dlasurvey is None):
        _, ml_dlasurvey = load_ml_dr7()
    # Load DR5
    dr5 = DLASurvey.load_SDSS_DR5()  # This is the statistical sample
    # Vette
    vette = ltu.loadjson(vette_file)
    dr5_ml_idx = np.array(vette['dr5_idx'])

    # Use coord to efficiently deal with sightlines
    ml_dla_coord = ml_dlasurvey.coords
    dr5_coord = SkyCoord(ra=dr5.sightlines['RA'], dec=dr5.sightlines['DEC'], unit='deg')
    idx, d2d, d3d = match_coordinates_sky(ml_dla_coord, dr5_coord, nthneighbor=1)
    in_dr5 = d2d < 2*u.arcsec
    print("{:d} of the ML DLA were in the DR5 sightlines".format(np.sum(in_dr5)))

    # False positives
    fpos = np.array([True]*ml_dlasurvey.nsys)
    fpos[~in_dr5] = False

    # False positives
    imatched = np.where(dr5_ml_idx >= 0)[0]
    match_val = dr5_ml_idx[imatched]
    fpos[match_val] = False
    print("There are {:d} total false positives".format(np.sum(fpos)))
    # This nearly matches David's.  Will run with his analysis.

    fpos_in_stat = fpos.copy()
    # Restrict on DR5
    plates = ml_dlasurvey.plate
    fibers = ml_dlasurvey.fiber
    zabs = ml_dlasurvey.zabs
    zem = ml_dlasurvey.zem
    for idx in np.where(fpos_in_stat)[0]:
        # Finally, match to DR5
        dr5_sl = np.where((dr5.sightlines['PLATE'] == plates[idx]) &
                          (dr5.sightlines['FIB'] == fibers[idx]))[0][0]
        if (zabs[idx] >= dr5.sightlines['Z_START'][dr5_sl]) & \
                (zabs[idx] <= dr5.sightlines['Z_END'][dr5_sl]):
            pass
        else:
            fpos_in_stat[idx] = False
    print("Number of FP in DR5 analysis region = {:d}".format(np.sum(fpos_in_stat)))
    print("Number with NHI<20.45 = {:d}".format(np.sum(ml_dlasurvey.NHI[fpos_in_stat]< 20.45)))

    # High NHI
    highNHI = ml_dlasurvey.NHI[fpos_in_stat] > 21.
    htbl = Table()
    htbl['PLATE'] = plates[fpos_in_stat][highNHI]
    htbl['FIBER'] = fibers[fpos_in_stat][highNHI]
    htbl['zabs'] = zabs[fpos_in_stat][highNHI]
    htbl['NHI'] = ml_dlasurvey.NHI[fpos_in_stat][highNHI]
    htbl.write("FP_DR5_highNHI.ascii", format='ascii.fixed_width', overwrite=True)

    # Medium NHI
    medNHI = (ml_dlasurvey.NHI[fpos_in_stat] > 20.6) & (ml_dlasurvey.NHI[fpos_in_stat] < 21)
    mtbl = Table()
    mtbl['PLATE'] = plates[fpos_in_stat][medNHI]
    mtbl['FIBER'] = fibers[fpos_in_stat][medNHI]
    mtbl['zabs'] = zabs[fpos_in_stat][medNHI]
    mtbl['zem'] = zem[fpos_in_stat][medNHI]
    mtbl['NHI'] = ml_dlasurvey.NHI[fpos_in_stat][medNHI]
    mtbl.write("FP_DR5_medNHI.ascii", format='ascii.fixed_width', overwrite=True)


def chk_pn_dla_to_ml(ml_dlasurvey=None, ml_llssurvey=None, dz_toler=0.015, outfile='vette_dr7_pn.json'):
    """ Compare results of Noterdaeme to ML
    Save to JSON file
    """
    # Load ML
    if (ml_dlasurvey is None) or (ml_llssurvey is None):
        ml_llssurvey, ml_dlasurvey = load_ml_dr7()
    # Load PN
    pn_dr7_file = '../Analysis/noterdaeme_dr7.fits'
    pn_dr7 = Table.read(pn_dr7_file)

    # Use coord to efficiently deal with sightlines
    ml_coord = SkyCoord(ra=ml_dlasurvey.sightlines['RA'], dec=ml_dlasurvey.sightlines['DEC'], unit='deg')
    pn_coord = SkyCoord(ra=pn_dr7['_RA'], dec=pn_dr7['_DE'], unit='deg')
    idx, d2d, d3d = match_coordinates_sky(pn_coord, ml_coord, nthneighbor=1)
    in_ml = d2d < 2*u.arcsec
    print("{:d} of the PN sightlines were covered by ML out of {:d}".format(np.sum(in_ml), len(pn_dr7)))

    # Cut
    cut_pn = pn_dr7[in_ml]

    # Loop on PN DLAs and save indices of the matches
    pn_ml_idx = np.zeros(len(cut_pn)).astype(int) - 1
    for ii,pnrow in enumerate(cut_pn):
        if pnrow['logN_HI_'] >= 20.3:
            dla_mts = np.where((ml_dlasurvey.plate == pnrow['Plate']) & (ml_dlasurvey.fiber == pnrow['Fiber']))[0]
            nmt = len(dla_mts)
            if nmt == 0:  # No match
                # Check for LLS
                lls_mts = np.where((ml_llssurvey.plate == pnrow['Plate']) & (ml_llssurvey.fiber == pnrow['Fiber']))[0]
                nmt2 = len(lls_mts)
                if nmt2 == 0:  # No match
                    pass
                else:
                    zML = ml_llssurvey.zabs[lls_mts] # Redshifts of all DLAs on the sightline in ML
                    zdiff = np.abs(pnrow['zabs']-zML)
                    if np.min(zdiff) < dz_toler:
                        pn_ml_idx[ii] = -9  # SLLS match
            else:
                zML = ml_dlasurvey.zabs[dla_mts] # Redshifts of all DLAs on the sightline in ML
                zdiff = np.abs(pnrow['zabs']-zML)
                if np.min(zdiff) < dz_toler:
                    #print("Match on {:d}!".format(ii))
                    # Match
                    imin = np.argmin(zdiff)
                    pn_ml_idx[ii] = dla_mts[imin]
        else:
            pn_ml_idx[ii] = -99  # Not a PN DLA
    # Stats on matches
    '''
    gdm = pn_ml_idx >= 0
    pdb.set_trace()
    dz = cut_pn['zabs'][gdm]-ml_dlasurvey.zabs[pn_ml_idx[gdm]]
    dNHI = cut_pn['logN_HI_'][gdm]-ml_dlasurvey.NHI[pn_ml_idx[gdm]]
    plt.clf()
    #plt.hist(dz)
    plt.hist(dNHI)
    plt.show()
    '''
    # PN not matched by ML?
    misses = (pn_ml_idx == -1)
    pn_missed = cut_pn[misses]
    # Write high NHI systems to disk
    high_NHI = pn_missed['logN_HI_'] > 20.8
    pn_missed[['QSO','Plate','Fiber', 'zem', 'zabs', 'Flag', 'logN_HI_']][high_NHI].write("N09_missed_highNHI.ascii", format='ascii.fixed_width', overwrite=True)

    # ML not matched by PN?
    ml_dla_coords = ml_dlasurvey.coords
    idx2, d2d2, d3d = match_coordinates_sky(ml_dla_coords, pn_coord, nthneighbor=1)
    not_in_pn = d2d2 > 2*u.arcsec  # This doesn't check zabs!!

    tmp_tbl = Table()
    for key in ['plate', 'fiber', 'zabs', 'NHI', 'confidence']:
        tmp_tbl[key] = getattr(ml_dlasurvey, key)

    # Save
    out_dict = {}
    out_dict['in_ml'] = in_ml
    out_dict['pn_idx'] = pn_ml_idx  # -1 are misses, -99 are not DLAs in PN
    out_dict['not_in_pn'] = np.where(not_in_pn)[0]
    ltu.savejson(outfile, ltu.jsonify(out_dict), overwrite=True)
    print("Wrote: {:s}".format(outfile))


def main(flg):

    if (flg & 2**0):  # Test load
        #profile()
        load_ml_dr7()

    if (flg & 2**1):  # Compare PN DLAs to ML
        chk_pn_dla_to_ml()

    if (flg & 2**2):  # Compare DR5 DLAs to ML
        chk_dr5_dla_to_ml()

    if (flg & 2**3):  # Compare DR5 DLAs to ML
        dr5_false_positives()


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1: #
        flg_vet = 0
        #flg_vet += 2**0   # Tests
        #flg_vet += 2**1   # Compare to N09
        #flg_vet += 2**2   # Compare to PW09
        flg_vet += 2**3   # False positives in DR5
    else:
        flg_vet = int(sys.argv[1])

    main(flg_vet)
