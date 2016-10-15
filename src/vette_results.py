""" Module to vette results against Human catalogs
  SDSS-DR5 (JXP) and BOSS (Notredaeme)
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import pdb

from astropy.table import Table
from linetools import utils as ltu
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

    """
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
                 classification=[])
    ml_tbl = Table()
    ml_survey = DLASurvey()
    systems = []
    in_ml = np.array([False]*len(sdss_survey.sightlines))
    # Loop
    for obj in ml_results:
        # Sightline
        for key in idict.keys():
            idict[key].append(obj[key])
        mt = np.where((sdss_survey.sightlines['PLATE'] == obj['plate']) & (
            sdss_survey.sightlines[fkey] == obj['fiber']))[0]
        # Save
        if len(mt) != 1:
            raise ValueError("Model sightline not in SDSS survey")
        in_ml[mt[0]] = True
        # DLAs
        for idla in obj['dlas']:
            """
            dla = DLASystem((sdss_survey.sightlines['RA'][mt[0]],
                             sdss_survey.sightlines['DEC'][mt[0]]),
                            idla['spectrum']/(1215.6701)-1., None,
                            idla['column_density'])
            """
            if idla['spectrum'] < 3800.:
                continue
            isys = LLSSystem((sdss_survey.sightlines['RA'][mt[0]],
                     sdss_survey.sightlines['DEC'][mt[0]]),
                    idla['spectrum']/(1215.6701)-1., None,
                    NHI=idla['column_density'])
            isys.confidence = idla['dla_confidence']
            pdb.set_trace()
            # Save
            systems.append(isys)
    # Finish
    ml_survey._abs_sys = systems
    ml_survey.sightlines = sdss_survey[in_ml]
    for key in idict.keys():
        ml_tbl[key] = idict[key]
    ml_survey.ml_tbl = ml_tbl
    pdb.set_trace()
    # Return
    return ml_survey


def main(flg_tst, sdss=None):

    # Load JSON for DR5
    if (flg_tst % 2**1) >= 2**0:
        if sdss is None:
            sdss = DLASurvey.load_SDSS_DR5()
        json_to_sdss_dlasurvey('../results/dr5_v1_predictions.json', sdss)

# Test
if __name__ == '__main__':
    flg_tst = 0
    flg_tst += 2**0   # Load JSON for DR5

    main(flg_tst)
