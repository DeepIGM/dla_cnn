class Sightline(object):

    def __init__(self, id, dlas=None, flux=None, loglam=None, z_qso=None):
        self.flux = flux
        self.loglam = loglam
        self.id = id
        self.dlas = dlas
        self.z_qso = z_qso


    # Returns the data in the legacy data1, qso_z format for code that hasn't been updated to the new format yet
    def get_legacy_data1_format(self):
        raw_data = {}
        raw_data['flux'] = self.flux
        raw_data['loglam'] = self.loglam
        raw_data['plate'] = self.id.plate if hasattr(self.id, 'plate') else 0
        raw_data['mjd'] = self.id.mjd if hasattr(self.id, 'mjd') else 0
        raw_data['fiber'] = self.id.fiber if hasattr(self.id, 'fiber') else 0
        raw_data['ra'] = self.id.ra if hasattr(self.id, 'ra') else 0
        raw_data['dec'] = self.id.dec if hasattr(self.id, 'dec') else 0
        return raw_data, self.z_qso

