""" Module for vetting 10k test run
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

def pred_to_tbl(pred_file):
    spec_list = ltu.loadjson(pred_file)
    ids, zabs, conf, NHI, sigNHI, biasNHI = [], [], [], [], [], []
    # Loop to my loop
    for ss,spec in enumerate(spec_list):
        for dla in spec['dlas']:
            if dla['type'] == "LYB":
                continue
            ids.append(ss)
            zabs.append(dla['z_dla'])
            NHI.append(dla['column_density'])
            sigNHI.append(dla['std_column_density'])
            biasNHI.append(dla['column_density_bias_adjust'])
            conf.append(dla['dla_confidence'])
    # Table
    dla_tbl = Table()
    dla_tbl['ids'] = ids
    dla_tbl['zabs'] = zabs
    dla_tbl['conf'] = conf
    dla_tbl['sigNHI'] = sigNHI
    dla_tbl['biasNHI'] = biasNHI
    dla_tbl['NHI'] = NHI
    # Return
    return dla_tbl


def test_to_tbl(test_file):
    test_dict = ltu.loadjson(test_file)
    ids, zabs, sl, NHI, = [], [], [], []
    ntest = len(test_dict)
    # Loop to my loop
    for ss in range(ntest):
        ndla = test_dict[str(ss)]['nDLA']
        for idla in range(ndla):
            ids.append(ss)
            zabs.append(test_dict[str(ss)][str(idla)]['zabs'])
            NHI.append(test_dict[str(ss)][str(idla)]['NHI'])
            sl.append(test_dict[str(ss)]['sl'])
    # Table
    test_tbl = Table()
    test_tbl['ids'] = ids
    test_tbl['zabs'] = zabs
    test_tbl['NHI'] = NHI
    test_tbl['sl'] = sl
    # Return
    return test_tbl


def score_ml_test(dz_toler=0.015, outfile='vette_10k.json',
                  test_file='data/test_dlas_96629_10000.json.gz',
                  pred_file='data/test_dlas_96629_predictions.json.gz'):
    # Load Test
    test_dlas = test_to_tbl(test_file)
    ntest = len(test_dlas)
    # Load ML
    ml_abs = pred_to_tbl(pred_file)


    # Loop on test DLAs and save indices of the matches
    test_ml_idx = np.zeros(ntest).astype(int) - 99999
    for ii in range(ntest):
        # Match to ML sl
        in_sl = np.where(ml_abs['ids'] == test_dlas['ids'][ii])[0]
        dla_mts = np.where(np.abs(ml_abs['zabs'][in_sl] - test_dlas['zabs'][ii]) < dz_toler)[0]
        nmt = len(dla_mts)
        if nmt == 0:  # No match within dz
            pass
        elif nmt == 1:  # No match
            if ml_abs['NHI'][in_sl][dla_mts[0]] > 20.2999:
                test_ml_idx[ii] = in_sl[dla_mts[0]]
            else:
                test_ml_idx[ii] = -1 * in_sl[dla_mts[0]]
        else:  # Very rarely the ML identifies two DLAs in the window
            print("Double hit in test DLA {:d}".format(ii))
            imin = np.argmin(np.abs(ml_abs['zabs'][in_sl] - test_dlas['zabs'][ii]))
            test_ml_idx[ii] = in_sl[imin]

    match = test_ml_idx >= 0
    print("There were {:d} DLAs recovered out of {:d}".format(np.sum(match), ntest))

    # Write out misses
    misses = np.where(test_ml_idx == -99999)[0]
    print("There were {:d} DLAs missed altogether".format(len(misses)))
    mtbl = Table()
    for key in ['sl', 'NHI', 'zabs']:
        mtbl[key] = test_dlas[key][misses]
    mtbl.write('test_misses.ascii', format='ascii.fixed_width', overwrite=True)

    # Write out SLLS
    sllss = np.where((test_ml_idx < 0) & (test_ml_idx != -99999))[0]
    print("There were {:d} DLAs recovered as SLLS".format(len(sllss)))
    stbl = Table()
    for key in ['sl', 'NHI', 'zabs']:
        stbl[key] = test_dlas[key][sllss]
    mtbl.write('test_slls.ascii', format='ascii.fixed_width', overwrite=True)

    # Save
    out_dict = {}
    out_dict['test_idx'] = test_ml_idx  # -1 are misses, -99 are not DLAs in PN, -9 are SLLS
    ltu.savejson(outfile, ltu.jsonify(out_dict), overwrite=True)

    # Stats on dz
    dz = ml_abs['zabs'][test_ml_idx[match]] - test_dlas['zabs'][match]
    print("Median dz = {} and sigma(dz)= {}".format(np.median(dz), np.std(dz)))


def examine_false_pos(test_file='data/test_dlas_96629_10000.json.gz',
                      pred_file='data/test_dlas_96629_predictions.json.gz',
                      vette_file='vette_10k.json'):
    """ Examine false positives in the Test set (held out)
    """
    from pyigm.surveys.dlasurvey import DLASurvey
    import h5py
    import json
    from matplotlib import pyplot as plt
    # Load Test
    test_dlas = test_to_tbl(test_file)
    ntest = len(test_dlas)
    # Load hdf5
    CNN_result_path = '/home/xavier/Projects/ML_DLA_results/CNN/'
    hdf5_datafile = CNN_result_path+'gensample_hdf5_files/test_dlas_96629_10000.hdf5'
    hdf = h5py.File(hdf5_datafile, 'r')
    headers = json.loads(hdf['meta'].value)['headers']
    # Load ML
    ml_abs = pred_to_tbl(pred_file)
    # Vette
    vette = ltu.loadjson(vette_file)
    test_ml_idx = np.array(vette['test_idx'])
    # Load DR5
    dr5 = DLASurvey.load_SDSS_DR5()
    all_dr5 = DLASurvey.load_SDSS_DR5(sample='all_sys')

    # False positives
    fpos = ml_abs['NHI'] >= 20.3  # Must be a DLA
    imatched = np.where(test_ml_idx >= 0)[0]
    match_val = test_ml_idx[imatched]
    fpos[match_val] = False
    print("There are {:d} total false positives".format(np.sum(fpos)))
    # This nearly matches David's.  Will run with his analysis.

    fpos_in_dr5 = fpos.copy()
    # Restrict on DR5
    for idx in np.where(fpos_in_dr5)[0]:
        # Convoluted indexing..
        mlid = ml_abs['ids'][idx]
        # Plate/Fiber
        plate = headers[mlid]['PLATE']
        fib = headers[mlid]['FIBER']
        # Finally, match to DR5
        dr5_sl = np.where((dr5.sightlines['PLATE'] == plate) &
                          (dr5.sightlines['FIB'] == fib))[0][0]
        if (ml_abs['zabs'][idx] >= dr5.sightlines['Z_START'][dr5_sl]) & \
                (ml_abs['zabs'][idx] <= dr5.sightlines['Z_END'][dr5_sl]):
            pass
        else:
            fpos_in_dr5[idx] = False
    print("Number of FP in DR5 analysis region = {:d}".format(np.sum(fpos_in_dr5)))

    # How many match to DR5 SLLS?
    slls = all_dr5.NHI < 20.3
    slls_coord = all_dr5.coord[slls]
    slls_zabs = all_dr5.zabs[slls]
    nslls = 0
    for idx in np.where(fpos_in_dr5)[0]:
        # Convoluted indexing..
        mlid = ml_abs['ids'][idx]
        # RA/DEC
        ra = headers[mlid]['RA_GROUP']
        dec = headers[mlid]['DEC_GROUP']
        coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        # Match coord
        mt = coord.separation(slls_coord) < 3*u.arcsec
        if np.any(mt):
            # Match redshift
            if np.min(np.abs(slls_zabs[mt] - ml_abs['zabs'][idx])) < 0.015:
                nslls += 1
    print("Number of FP that are SLLS in DR5 = {:d}".format(nslls))

    low_NHI = ml_abs['NHI'][fpos_in_dr5] < 20.5
    print("Number of FP that are NHI <= 20.5 = {:d}".format(np.sum(low_NHI)))

    # Write out
    fp_tbl = Table()
    for key in ['ids', 'NHI', 'zabs', 'conf']:
        fp_tbl[key] = ml_abs[key][fpos_in_dr5]
    fp_tbl.write('test10k_false_pos.ascii', format='ascii.fixed_width', overwrite=True)

    # Histogram
    dr5_idx = np.where(fpos_in_dr5)
    plt.clf()
    ax = plt.gca()
    ax.hist(ml_abs['conf'][dr5_idx])
    plt.show()


def high_nhi_neg():
    """ Examine High NHI false negatives in 10k test
    """
    # Load ML
    ml_abs = pred_to_tbl('../Vetting/data/test_dlas_96629_predictions.json.gz')
    # Load Test
    test_dlas = test_to_tbl('../Vetting/data/test_dlas_96629_10000.json.gz')
    # Load vette
    vette_10k = ltu.loadjson('../Vetting/vette_10k.json')
    test_ml_idx = np.array(vette_10k['test_idx'])

    misses = np.where(test_ml_idx == -99999)[0]
    highNHI = test_dlas['NHI'][misses] > 21.2
    high_tbl = test_dlas[misses[highNHI]]

    # Write
    high_tbl.write('test_highNHI_neg.ascii', format='ascii.fixed_width', overwrite=True)

def main(flg):

    if (flg & 2**0):  # Scorecard the ML run
        score_ml_test() # 10k
        #score_ml_test(outfile='vette_5k.json',
        #          test_file='data/test_dlas_5k96451.json.gz',
        #          pred_file='data/test_dlas_5k96451_predictions.json.gz')

    if (flg & 2**1):  # Generate list of high NHI
        high_nhi_neg()

    if (flg & 2**2):  # Generate list of high NHI
        examine_false_pos()

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1: #
        flg_vet = 0
        #flg_vet += 2**0   # Main scorecard
        #flg_vet += 2**1   # High NHI
        flg_vet += 2**2   # False positives
    else:
        flg_vet = int(sys.argv[1])

    main(flg_vet)
