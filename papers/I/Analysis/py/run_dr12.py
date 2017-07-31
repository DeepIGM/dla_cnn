""" Module for analyzing metallicty with MCMC
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import sys
import io, json
import pdb

from pkg_resources import resource_filename

from astropy import units as u
from astropy.table import Table, Column

from specdb.specdb import IgmSpec

from dla_cnn.data_loader import process_catalog_dr12

def generate_csv():
    igmsp = IgmSpec()
    boss_dr12 = igmsp['BOSS_DR12'].meta
    gdq = boss_dr12['zem_GROUP'] > 1.95
    boss_dr12 = boss_dr12[gdq]
    # Build the Table -- NOTE THE ORDER DOES MATTER!
    dr12_set = Table()
    dr12_set['PLATE'] = boss_dr12['PLATE']
    dr12_set['MJD'] = boss_dr12['MJD']
    dr12_set['FIB'] = boss_dr12['FIBERID']
    dr12_set['RA'] = boss_dr12['RA_GROUP']
    dr12_set['DEC'] = boss_dr12['DEC_GROUP']
    # Write
    dr12_set.write('dr12_set.csv', format='csv', overwrite=True)

def process_dr12():
    # Set data model
    default_model = resource_filename('dla_cnn', "models/model_gensample_v7.1")
    # Run
    process_catalog_dr12(csv_plate_mjd_fiber="./dr12_set.csv",
                        kernel_size=400,
                        model_checkpoint=default_model,
                        output_dir="./visuals_dr12")

def main(flg):

    if (flg & 2**0):
        generate_csv()

    if (flg & 2**1):
        process_dr12()


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1: #
        flg_analy = 0
        #flg_analy += 2**0   # CSV
        flg_analy += 2**1   # Run on DR12
    else:
        flg_analy = int(sys.argv[1])

    main(flg_analy)
