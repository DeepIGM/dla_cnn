""" Module for tables of DLA CNN paper
"""

# Imports
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob, os, sys
import warnings
import pdb

import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from pkg_resources import resource_filename

from scipy.optimize import minimize
from scipy.stats import chisquare

from astropy import units as u
from astropy.table import Table

from linetools import utils as ltu

from pyigm.surveys.dlasurvey import DLASurvey, dla_stat


from dla_cnn.io import load_ml_dr7


# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#sys.path.append(os.path.abspath("../Vetting/py"))
#from vette_dr7 import load_ml_dr7

default_model = resource_filename('dla_cnn', "models/model_gensample_v7.1")

def init_for_ipython():
    #from imp import reload
    #import paperI_figs as pfigs
    _, ml_dlasurvey = load_ml_dr7()
    return ml_dlasurvey

# Summary table of DR7 DLAs
def mktab_dr7(outfil='tab_dr7_dlas.tex', ml_dlasurvey=None, sub=False):

    # Load DLA samples
    if ml_dlasurvey is None:
        _, ml_dlasurvey = load_ml_dr7()

    # Load DR7 vette file
    vette_file = '../Vetting/vette_dr7_pn.json'
    vdr7 = ltu.loadjson(vette_file)
    in_ml = np.array(vdr7['in_ml'])
    pn_ml_idx = np.array(vdr7['pn_idx'])
    not_in_pn = np.array(vdr7['not_in_pn'])
    #print("There are {:d} DLAs in ML and not in N09".format(len(not_in_pn)))
    ml_in_pn = np.array([True]*len(ml_dlasurvey._abs_sys))
    ml_in_pn[not_in_pn] = False

    # Shen (for BALs)
    shen = Table.read('dr7_bh_Nov19_2013.fits.gz')

    # Open
    tbfil = open(outfil, 'w')

    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\begin{minipage}{170mm} \n')
    tbfil.write('\\caption{SDSS DR7 DLA CANDIDATES\\label{tab:dr7}}\n')
    tbfil.write('\\begin{tabular}{lccccccc}\n')
    tbfil.write('\\hline \n')
    #tbfil.write('\\rotate\n')
    #tbfil.write('\\tablewidth{0pc}\n')
    #tbfil.write('\\tabletypesize{\\small}\n')
    tbfil.write('Plate & Fiber & \\zabs & NHI & Conf. & BAL \n')
    tbfil.write('& N09')
    tbfil.write('\\\\ \n')
    #tbfil.write('& & & (\AA) & (10$^{-15}$) & & (10$^{-17}$) &  ')
    #tbfil.write('} \n')
    tbfil.write('\\hline \n')

    #tbfil.write('\\startdata \n')

    bals, N09 = [], []
    for ii,dla in enumerate(ml_dlasurvey._abs_sys):
        # Match to shen
        mt_shen = np.where( (shen['PLATE'] == dla.plate) & (shen['FIBER'] == dla.fiber))[0]
        if len(mt_shen) != 1:
            pdb.set_trace()
        # Generate line
        dlac = '{:d} & {:d} & {:0.3f} & {:0.2f} & {:0.2f} & {:d}'.format(
            dla.plate, dla.fiber, dla.zabs, dla.NHI, dla.confidence, shen['BAL_FLAG'][mt_shen[0]])
        bals.append(shen['BAL_FLAG'][mt_shen[0]])
        # Add N09
        if ml_in_pn[ii]:
            dlac += '& 1'
            N09.append(1)
        else:
            dlac += '& 0'
            N09.append(0)
        # End line
        tbfil.write(dlac)
        tbfil.write('\\\\ \n')

    # Some stats for the paper
    gd_conf = ml_dlasurvey.confidence > 0.9
    gd_BAL = np.array(bals) == 0
    gd_z = ml_dlasurvey.zabs < ml_dlasurvey.zem
    new = np.array(N09) == 0
    gd_new = gd_BAL & gd_conf & new & gd_z

    new_dlas = Table()
    new_dlas['PLATE'] = ml_dlasurvey.plate[gd_new]
    new_dlas['FIBER'] = ml_dlasurvey.fiber[gd_new]
    new_dlas['zabs'] = ml_dlasurvey.zabs[gd_new]
    new_dlas['NHI'] =  ml_dlasurvey.NHI[gd_new]
    print("There are {:d} DR7 candidates.".format(ml_dlasurvey.nsys))
    print("There are {:d} DR7 candidates not in BAL.".format(np.sum(gd_BAL)))
    print("There are {:d} good DR7 candidates not in BAL.".format(np.sum(gd_BAL&gd_conf)))
    print("There are {:d} good DR7 candidates not in N09 nor BAL".format(np.sum(gd_new)))

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    tbfil.write('\\end{minipage} \n')
    #tbfil.write('{$^a$}Rest-frame value.  Error is dominated by uncertainty in $n_e$.\\\\ \n')
    #tbfil.write('{$^b$}Assumes $\\nu=1$GHz, $n_e = 4 \\times 10^{-3} \\cm{-3}$, $z_{\\rm DLA} = 1$, $z_{\\rm source} = 2$.\\\\ \n')
    tbfil.write('\\end{table*} \n')

    #tbfil.write('\\enddata \n')
    #tbfil.write('\\tablenotetext{a}{Flag describing the continuum method applied: 0=Analysis based only on Lyman series lines; 1=Linear fit; 2=Constant fit; 3=Continuum imposed by hand.}\n')
    #tbfil.write('\\tablecomments{Units for $C_0$ and $C_1$ are erg/s/cm$^2$/\\AA\ and erg/s/cm$^2$/\\AA$^2$ respecitvely.}\n')
    # End
    #tbfil.write('\\end{deluxetable*} \n')

    tbfil.close()
    print('Wrote {:s}'.format(outfil))

#### ########################## #########################
def main(flg_tab):

    if flg_tab == 'all':
        flg_tab = np.sum( np.array( [2**ii for ii in range(5)] ))
    else:
        flg_tab = int(flg_tab)

    # Compare dz and dNHI between N07 and ML
    if flg_tab & (2**0):
        mktab_dr7()

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_tab = 0
        flg_tab += 2**0   # DR7
    else:
        flg_tab = sys.argv[1]

    main(flg_tab)
