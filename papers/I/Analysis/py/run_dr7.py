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

from dla_cnn.data_loader import process_catalog_dr7

def tests():
    from dla_cnn.data_loader import read_igmspec
    raw_data, zqso = read_igmspec(750, 82)
    pdb.set_trace()

def process_dr7():
    # Set data model
    default_model = resource_filename('dla_cnn', "models/model_gensample_v7.1")
    # Run
    process_catalog_dr7(csv_plate_mjd_fiber="./dr7_set.csv",
                        kernel_size=400,
                        model_checkpoint=default_model,
                        output_dir="./visuals_dr7")

def main(flg):

    if (flg & 2**0):
        tests()

    if (flg & 2**1):
        process_dr7()


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1: #
        flg_analy = 0
        #flg_analy += 2**0   # Tests
        flg_analy += 2**1   # Run on DR7
    else:
        flg_analy = int(sys.argv[1])

    main(flg_analy)
