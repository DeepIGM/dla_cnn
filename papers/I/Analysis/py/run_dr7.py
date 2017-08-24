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

from dla_cnn.data_loader import process_catalog_dr7, add_s2n_after

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


def add_s2n(outfile='visuals_dr7/predictions_SDSSDR7_s2n.json'):
    from dla_cnn.data_model.Id_DR7 import Id_DR7

    csv_plate_mjd_fiber="./dr7_set.csv"
    csv = Table.read(csv_plate_mjd_fiber)
    ids = [Id_DR7(c[0],c[1],c[2],c[3]) for c in csv]
    jfile = 'visuals_dr7/predictions_SDSSDR7.json'
    # Call
    predictions = add_s2n_after(ids, jfile, CHUNK_SIZE=600)

    # Write JSON string
    with open(outfile, 'w') as f:
        json.dump(predictions, f, indent=4)


def main(flg):

    if (flg & 2**0):
        tests()

    if (flg & 2**1):
        process_dr7()

    if (flg & 2**2):
        add_s2n()


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1: #
        flg_analy = 0
        #flg_analy += 2**0   # Tests
        #flg_analy += 2**1   # Run on DR7
        flg_analy += 2**2   # Add S/N after the fact
    else:
        flg_analy = int(sys.argv[1])

    main(flg_analy)
