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
    ids, zabs, conf, NHI, sigNHI = [], [], [], [], []
    # Loop to my loop
    for ss,spec in enumerate(spec_list):
        for dla in spec['dlas']:
            ids.append(ss)
            zabs.append(dla['z_dla'])
            NHI.append(dla['column_density'])
            sigNHI.append(dla['std_column_density'])
            conf.append(dla['dla_confidence'])
    # Table
    dla_tbl = Table()
    dla_tbl['ids'] = ids
    dla_tbl['zabs'] = zabs
    dla_tbl['conf'] = conf
    dla_tbl['sigNHI'] = sigNHI
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


def score_ml_10ktest(ml_dlasurvey=None, ml_llssurvey=None, dz_toler=0.015,
                      outfile='vette_10k.json'):
    # Load ML
    ml_abs = pred_to_tbl('data/test_dlas_96629_predictions.json.gz')
    # Load Test
    test_dlas = test_to_tbl('data/test_dlas_96629_10000.json.gz')
    ntest = len(test_dlas)


    # Loop on test DLAs and save indices of the matches
    test_ml_idx = np.zeros(ntest).astype(int) - 1
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
                test_ml_idx[ii] = -9
        else:  # Very rarely the ML identifies two DLAs in the window
            print("Double hit in test DLA {:d}".format(ii))
            imin = np.argmin(np.abs(ml_abs['zabs'][in_sl] - test_dlas['zabs'][ii]))
            test_ml_idx[ii] = in_sl[imin]

    match = test_ml_idx >= 0
    print("There were {:d} DLAs recovered out of {:d}".format(np.sum(match), ntest))

    # Write out misses
    misses = np.where(test_ml_idx == -1)[0]
    print("There were {:d} DLAs missed altogether".format(len(misses)))
    mtbl = Table()
    for key in ['sl', 'NHI', 'zabs']:
        mtbl[key] = test_dlas[key][misses]
    mtbl.write('test_misses.ascii', format='ascii.fixed_width', overwrite=True)

    # Write out SLLS
    sllss = np.where(test_ml_idx == -9)[0]
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


def main(flg):

    if (flg & 2**0):  # Scorecard the ML run
        score_ml_10ktest()

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1: #
        flg_vet = 0
        flg_vet += 2**0   # Main scorecard
        #flg_vet += 2**1   # Run on DR7
        #flg_vet += 2**2   # Run on DR7
    else:
        flg_vet = int(sys.argv[1])

    main(flg_vet)
