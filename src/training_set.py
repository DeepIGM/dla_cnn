""" Module to vette results against Human catalogs
  SDSS-DR5 (JXP) and BOSS (Notredaeme)
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import pdb


#from astropy.table import Table
#from astropy.coordinates import SkyCoord, match_coordinates_sky
#from astropy import units as u

from linetools import utils as ltu
from pyigm.surveys.dlasurvey import DLASurvey


#def grab_clean_sightlines():
