#!/usr/bin/env python

"""
Generate a training set by script
"""
from __future__ import (print_function, absolute_import, division, unicode_literals)

import pdb

try:
    ustr = unicode
except NameError:
    ustr = str

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(
        description='Print coordinates in several formats from input one.')
    parser.add_argument("seed", type=int, help="Seed for random number generation")
    parser.add_argument("ntrain", type=int, help="Number of training sightlines to generate")
    parser.add_argument("outpath", type=str, help="Output path")
    parser.add_argument("--slls", default=False, action="store_true", help="Generate only SLLS?")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def main(args=None):
    pargs = parser(options=args)

    # Setup
    import sys
    pfind = __file__.rfind('/scripts')
    spth = __file__[:pfind]+'/src'
    sys.path.append(spth)
    import training_set as tset
    from pyigm.surveys.dlasurvey import DLASurvey

    outroot = pargs.outpath+'/training_{:d}_{:d}'.format(pargs.seed, pargs.ntrain)

    # Sightlines
    sdss = DLASurvey.load_SDSS_DR5(sample='all')
    slines, sdict = tset.grab_sightlines(sdss, flg_bal=0)
    # Run
    _, _ = tset.make_set(pargs.ntrain, slines, outroot=outroot, seed=pargs.seed, slls=pargs.slls)


if __name__ == '__main__':
    main()
