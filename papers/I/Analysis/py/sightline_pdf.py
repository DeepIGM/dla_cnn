#!/usr/bin/env python

"""
Script to generate a PDF of desired sightline
"""

import pdb

try:
    ustr = unicode
except NameError:
    ustr = str


def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(
        description='Generate a PDF of the desired sightline (v1.0)')
    parser.add_argument("plate", type=int, help="Plate")
    parser.add_argument("fiber", type=int, help="Fiber")
    parser.add_argument("survey", type=str, help="SDSS_DR7, BOSS_DR12")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args=None):
    from pkg_resources import resource_filename
    from dla_cnn.data_loader import process_catalog_dr7

    if args is None:
        pargs = parser()
    else:
        pargs = args
    default_model = resource_filename('dla_cnn', "models/model_gensample_v7.1")
    if pargs.survey == 'SDSS_DR7':
        cvs_file = resource_filename('dla_cnn', "catalogs/sdss_dr7/dr7_set.csv")
        process_catalog_dr7(csv_plate_mjd_fiber=cvs_file,
                        kernel_size=400, model_checkpoint=default_model,
                        output_dir="./", pfiber=(pargs.plate, pargs.fiber),
                        make_pdf=True)

    # Run sightline
    path = './'

# Command line execution
if __name__ == '__main__':
    args = parser()
    main(args)
