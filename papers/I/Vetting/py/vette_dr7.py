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

def load_ml_dr7():
    # Read
    ml_results = ltu.loadjson(dr7_file)
    use_platef = False
    if 'plate' in ml_results[0].keys():
        use_platef = True
    else:
        if 'id' in ml_results[0].keys():
            use_id = True
    # Init
    #idict = dict(plate=[], fiber=[], classification_confidence=[],  # FOR v2
    #             classification=[], ra=[], dec=[])
    idict = dict(ra=[], dec=[], plate=[], fiber=[])
    if use_platef:
        for key in ['plate', 'fiber', 'mjd']:
            idict[key] = []
    dlasystems = []
    llssystems = []

    # Generate coords to speed things up
    for obj in ml_results:
        for key in ['ra', 'dec']:
            idict[key].append(obj[key])
    ml_coords = SkyCoord(ra=idict['ra'], dec=idict['dec'], unit='deg')
    ra_names = ml_coords.icrs.ra.to_string(unit=u.hour,sep='',pad=True)
    dec_names = ml_coords.icrs.dec.to_string(sep='',pad=True,alwayssign=True)
    vlim = [-500., 500.]*u.km/u.s
    dcoord = SkyCoord(ra=0., dec=0., unit='deg')

    # Loop on list
    didx, lidx = [], []
    print("Looping on sightlines..")
    for tt,obj in enumerate(ml_results):
        #if (tt % 100) == 0:
        #    print('tt: {:d}'.format(tt))
        # Sightline
        if use_id:
            plate, fiber = [int(spl) for spl in obj['id'].split('-')]
            idict['plate'].append(plate)
            idict['fiber'].append(fiber)

        # Systems
        for ss,syskey in enumerate(['dlas', 'subdlas']):
            for idla in obj[syskey]:
                name = 'J{:s}{:s}_z{:.3f}'.format(ra_names[tt], dec_names[tt], idla['z_dla'])
                if ss == 0:
                    isys = DLASystem(dcoord, idla['z_dla'], vlim, NHI=idla['column_density'], zem=obj['z_qso'], name=name)
                else:
                    isys = LLSSystem(dcoord, idla['z_dla'], vlim, NHI=idla['column_density'], zem=obj['z_qso'], name=name)
                isys.confidence = idla['dla_confidence']
                if use_platef:
                    isys.plate = obj['plate']
                    isys.fiber = obj['fiber']
                elif use_id:
                    isys.plate = plate
                    isys.fiber = fiber
                # Save
                if ss == 0:
                    didx.append(tt)
                    dlasystems.append(isys)
                else:
                    lidx.append(tt)
                    llssystems.append(isys)
    # Generate sightline tables
    sightlines = Table()
    sightlines['RA'] = idict['ra']
    sightlines['DEC'] = idict['dec']
    sightlines['PLATE'] = idict['plate']
    sightlines['FIBERID'] = idict['fiber']
    # Surveys
    ml_llssurvey = LLSSurvey()
    ml_llssurvey.sightlines = sightlines.copy()
    ml_llssurvey._abs_sys = llssystems
    ml_llssurvey.coords = ml_coords[np.array(lidx)]

    ml_dlasurvey = DLASurvey()
    ml_dlasurvey.sightlines = sightlines.copy()
    ml_dlasurvey._abs_sys = dlasystems
    ml_dlasurvey.coords = ml_coords[np.array(didx)]

    # Return
    return ml_llssurvey, ml_dlasurvey

def chk_dr5_dla_to_ml(ml_dlasurvey=None, ml_llssurvey=None, dz_toler=0.03,
                      outfile='vette_dr5.json'):
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

    # Save
    out_dict = {}
    out_dict['in_ml'] = in_ml
    out_dict['dr5_idx'] = dr5_ml_idx  # -1 are misses, -99 are not DLAs in PN, -9 are SLLS
    #out_dict['not_in_pn'] = np.where(not_in_pn)[0]
    ltu.savejson(outfile, ltu.jsonify(out_dict), overwrite=True)


def chk_pn_dla_to_ml(ml_dlasurvey=None, ml_llssurvey=None, dz_toler=0.03, outfile='vette_dr7_pn.json'):
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

    # ML not matched by PN?
    ml_dla_coords = ml_dlasurvey.coords
    idx2, d2d2, d3d = match_coordinates_sky(ml_dla_coords, pn_coord, nthneighbor=1)
    not_in_pn = d2d2 > 2*u.arcsec

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



# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1: #
        flg_vet = 0
        #flg_vet += 2**0   # Tests
        #flg_vet += 2**1   # Run on DR7
        flg_vet += 2**2   # Run on DR7
    else:
        flg_vet = int(sys.argv[1])

    main(flg_vet)
