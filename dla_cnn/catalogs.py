""" Module of routines related to DLA catalogs
However I/O is in io.py
"""
from __future__ import print_function, absolute_import, division, unicode_literals

from pkg_resources import resource_filename

from astropy.table import Table

from linetools import utils as ltu

def generate_boss_tables():
    """
    Returns
    -------

    """
    # Load JSON file
    dr12_json = resource_filename('dla_cnn', 'catalogs/boss_dr12/predictions_DR12.json')
    dr12 = ltu.loadjson(dr12_json)

    # Parse into tables
    s_plates = []
    s_fibers = []
    s_mjds = []
    s_ra = []
    s_dec = []
    s_zem = []

    a_zabs = []
    a_NHI = []
    a_sigNHI = []
    a_conf = []
    a_plates = []
    a_fibers = []
    a_mjds = []
    a_ra = []
    a_dec = []
    a_zem = []
    for sline in dr12:
        # Plate/fiber
        plate, mjd, fiber = [int(spl) for spl in sline['id'].split('-')]
        s_plates.append(plate)
        s_mjds.append(mjd)
        s_fibers.append(fiber)
        # RA/DEC/zem
        s_ra.append(sline['ra'])
        s_dec.append(sline['dec'])
        s_zem.append(sline['z_qso'])
        # DLAs/SLLS
        for abs in sline['dlas']+sline['subdlas']:
            a_plates.append(plate)
            a_mjds.append(mjd)
            a_fibers.append(fiber)
            # RA/DEC/zem
            a_ra.append(sline['ra'])
            a_dec.append(sline['dec'])
            a_zem.append(sline['z_qso'])
            # Absorber
            a_zabs.append(abs['z_dla'])
            a_NHI.append(abs['column_density'])
            a_sigNHI.append(abs['std_column_density'])
            a_conf.append(abs['dla_confidence'])
    # Sightline tables
    sline_tbl = Table()
    sline_tbl['Plate'] = s_plates
    sline_tbl['Fiber'] = s_fibers
    sline_tbl['MJD'] = s_mjds
    sline_tbl['RA'] = s_ra
    sline_tbl['DEC'] = s_dec
    sline_tbl['zem'] = s_zem
    dr12_sline = resource_filename('dla_cnn', 'catalogs/boss_dr12/DR12_sightlines.fits')
    sline_tbl.write(dr12_sline, overwrite=True)
    # DLA/SLLS table
    abs_tbl = Table()
    abs_tbl['Plate'] = a_plates
    abs_tbl['Fiber'] = a_fibers
    abs_tbl['MJD'] = a_mjds
    abs_tbl['RA'] = a_ra
    abs_tbl['DEC'] = a_dec
    abs_tbl['zem'] = a_zem
    #
    abs_tbl['zabs'] = a_zabs
    abs_tbl['NHI'] = a_NHI
    abs_tbl['sigNHI'] = a_sigNHI
    abs_tbl['conf'] = a_conf
    dr12_abs = resource_filename('dla_cnn', 'catalogs/boss_dr12/DR12_DLA_SLLS.fits')
    abs_tbl.write(dr12_abs, overwrite=True)


def main(flg_cat):
    import os

    # BOSS tables
    if (flg_cat & 2**0):
        generate_boss_tables()

# Test
if __name__ == '__main__':
    flg_cat = 0
    flg_cat += 2**0   # BOSS Tables
    #flg_cat += 2**7   # Training set of high NHI systems

    main(flg_cat)
