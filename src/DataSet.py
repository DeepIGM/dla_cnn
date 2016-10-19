import numpy as np

class DataSet:
    def __init__(self, raw_data):
        """Construct a DataSet"""
        self._fluxes = raw_data[:, :-8]
        self._labels = raw_data[:, -1]
        self._col_density = raw_data[:, -2]
        self._central_wavelength = raw_data[:, -3]
        self._plate = raw_data[:, -4]
        self._mjd = raw_data[:, -5]
        self._fiber = raw_data[:, -6]
        self._ra = raw_data[:, -7]
        self._dec = raw_data[:, -8]

        self._fluxes[np.isnan(self._fluxes)] = 0  # TODO change this to interpolate

        self._samples_consumed = 0
        self._ix_permutation = np.random.permutation(np.shape(self._labels)[0])

    @property
    def loglam(self):
        return self._loglam

    @property
    def fluxes(self):
        return self._fluxes

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

    def next_batch(self, batch_size):
        batch_ix = self._ix_permutation[0:batch_size]
        self._ix_permutation = np.roll(self._ix_permutation, batch_size)

        # keep track of how many samples have been consumed and reshuffle after an epoch has elapsed
        self._samples_consumed += batch_size
        if self._samples_consumed > np.shape(self._labels)[0]:
            self._ix_permutation = np.random.permutation(np.shape(self._labels)[0])
            self._samples_consumed = 0

        return self._fluxes[batch_ix, :], self._labels[batch_ix], self._col_density[batch_ix]

