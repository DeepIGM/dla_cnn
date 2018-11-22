""" Data object for SDSS DR7.  Uses IGMSPEC for the data """

from pkg_resources import resource_filename
import pdb

import numpy as np

from astropy.table import Table

from specdb.specdb import IgmSpec
from specdb import cat_utils

from dla_cnn.data_model import Data
from dla_cnn.data_model import Id
from dla_cnn.data_model import Sightline
from dla_cnn.data_model import data_utils

class SDSSDR7(Data.Data):

    def __init__(self, catalog_file=None):
        super(SDSSDR7,self).__init__()

        # Load IGMSpec which holds our data
        self.igmsp = IgmSpec()
        self.meta = self.igmsp['SDSS_DR7'].meta

        # Catalog
        if catalog_file is None:
            self.catalog_file = resource_filename('dla_cnn', 'catalogs/sdss_dr7/dr7_set.csv')
        self.load_catalog()

    def gen_ID(self, plate, fiber, ra=None, dec=None, group_id=-1):
        return Id_DR7(plate, fiber, ra=ra, dec=dec, group_id=group_id)

    def load_catalog(self, csv=True):
        # Load it
        self.catalog = Table.read(self.catalog_file)
        # Add IDs
        meta_pfib = self.meta['PLATE']*1000000 + self.meta['FIBER']
        cat_pfib = self.catalog['PLATE']*1000000 + self.catalog['FIBER']
        mIDs = cat_utils.match_ids(cat_pfib, meta_pfib, require_in_match=False)
        self.catalog['GROUP_ID'] = mIDs

    def load_IDs(self, pfiber=None):
        ids = [self.gen_ID(c['PLATE'],c['FIBER'],ra=c['RA'],dec=c['DEC'],
                           group_id=c['GROUP_ID']) for ii,c in enumerate(self.catalog)]
        # Single ID using plate/fiber?
        if pfiber is not None:
            plates = np.array([iid.plate for iid in ids])
            fibers = np.array([iid.fiber for iid in ids])
            imt = np.where((plates==pfiber[0]) & (fibers==pfiber[1]))[0]
            if len(imt) != 1:
                print("Plate/Fiber not in DR7!!")
                pdb.set_trace()
            else:
                ids = [ids[imt[0]]]
        return ids

    def load_data(self, id):
        data, meta = self.igmsp['SDSS_DR7'].grab_specmeta(id.group_id, use_XSpec=False)
        z_qso = meta['zem_GROUP'][0]

        flux = data['flux'].flatten() #np.array(spec[0].flux)
        sig = data['sig'].flatten() # np.array(spec[0].sig)
        loglam = np.log10(data['wave'].flatten())

        gdi = np.isfinite(loglam)

        (loglam_padded, flux_padded, sig_padded) = data_utils.pad_loglam_flux(
            loglam[gdi], flux[gdi], z_qso, sig=sig[gdi]) # Sanity check that we're getting the log10 values
        assert np.all(loglam < 10), "Loglam values > 10, example: %f" % loglam[0]

        raw_data = {}
        raw_data['flux'] = flux_padded
        raw_data['sig'] = sig_padded
        raw_data['loglam'] = loglam_padded
        raw_data['plate'] = id.plate
        raw_data['mjd'] = 0
        raw_data['fiber'] = id.fiber
        raw_data['ra'] = id.ra
        raw_data['dec'] = id.dec
        assert np.shape(raw_data['flux']) == np.shape(raw_data['loglam'])
        #sys.stdout = stdout
        # Return
        return raw_data, z_qso

    def read_sightline(self, id):
        sightline = Sightline.Sightline(id=id)
        # Data
        data1, z_qso = self.load_data(id)
        # Fill
        sightline.id.ra = data1['ra']
        sightline.id.dec = data1['dec']
        sightline.flux = data1['flux']
        sightline.sig = data1['sig']
        sightline.loglam = data1['loglam']
        sightline.z_qso = z_qso
        # Giddy up
        return sightline

class Id_DR7(Id.Id):
    def __init__(self, plate, fiber, ra=0, dec=0, group_id=-1):
        super(Id_DR7,self).__init__()
        self.plate = plate
        self.fiber = fiber
        self.ra = ra
        self.dec = dec
        self.group_id = group_id

    def id_string(self):
        return "%05d-%05d" % (self.plate, self.fiber)


def process_catalog_dr7(kernel_size=400, pfiber=None, make_pdf=False,
                        model_checkpoint=None, #default_model,
                        output_dir="../tmp/visuals_dr7",
                        debug=True):
    """ Generates a SDSS DR7 DLA catalog from plate/mjd/fiber from a CSV file
    Parameters
    ----------
    csv_plate_mjd_fiber
    kernel_size
    pfiber
    make_pdf
    model_checkpoint
    output_dir

    Returns
    -------

    """
    from dla_cnn.data_loader import process_catalog
    #
    data = SDSSDR7()
    ids = data.load_IDs(pfiber=pfiber)
    #
    process_catalog(ids, kernel_size, model_checkpoint, make_pdf=make_pdf,
                    CHUNK_SIZE=500, output_dir=output_dir, data=data, debug=debug)
