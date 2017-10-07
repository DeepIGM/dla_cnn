""" Module for analyzing metallicty with MCMC
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import sys
import os
import io, json
import pdb

from pkg_resources import resource_filename

from dla_cnn.data_loader import process_catalog_gensample


def process_lows2n():
    # Set data model
    default_model = resource_filename('dla_cnn', "models/model_gensample_v7.1")
    # Run
    hdf5_file = os.getenv('DROPBOX_DIR')+'/MachineLearning/LowS2N/lows2n_train_*_10000.hdf5'
    json_file = os.getenv('DROPBOX_DIR')+'/MachineLearning/LowS2N/lows2n_train_*_10000.json'
    process_catalog_gensample(gensample_files_glob=hdf5_file,
                              json_files_glob=json_file,
                              kernel_size=400,
                              model_checkpoint=default_model,
                              output_dir="visuals_lows2n/", debug=False)

def main(flg):

    if (flg & 2**0):
        tests()

    if (flg & 2**1):
        process_lows2n()


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1: #
        flg_analy = 0
        #flg_analy += 2**0   # Tests
        flg_analy += 2**1   # Run
    else:
        flg_analy = int(sys.argv[1])

    main(flg_analy)
