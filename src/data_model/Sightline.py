class Sightline:

    def __init__(self, raw_data):
        None


    @property
    def fluxes(self):
        return self._fluxes


    @property
    def loglam(self):
        return self._loglam


    @property
    def labels(self):
        return self._labels


    @property
    def col_density(self):
        return self._col_density


    @property
    def plate(self):
        return self._plate


    @property
    def mjd(self):
        return self._mjd


    @property
    def fiber(self):
        return self._fiber


    @property
    def ra(self):
        return self._ra


    @property
    def dec(self):
        return self.dec

    def scan_flux_samples(self):
        None