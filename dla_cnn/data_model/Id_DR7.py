from Id import Id

from pkg_resources import resource_filename

class Id_DR7(Id):
    @classmethod
    def from_csv(cls, plate, fiber):
        from astropy.table import Table
        import numpy as np
        csv_file = resource_filename('dla_cnn', 'catalogs/sdss_dr7/dr7_set.csv')
        csv = Table.read(csv_file)
        # Match
        idx = np.where((csv['PLATE']==plate) & (csv['FIB']==fiber))[0]
        if len(idx) == 0:
            raise IOError("Bad plate/fiber for SDSS DR7")
        # Init
        id_dr7 = cls(plate, fiber, ra=csv['RA'][idx[0]], dec=csv['DEC'][idx[0]])
        # Return
        return id_dr7

    def __init__(self, plate, fiber, ra=0, dec=0):
        super(Id_DR7,self).__init__()
        self.plate = plate
        self.fiber = fiber
        self.ra = ra
        self.dec = dec

    def id_string(self):
        return "%05d-%05d" % (self.plate, self.fiber)


