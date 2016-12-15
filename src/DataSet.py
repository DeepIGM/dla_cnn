import numpy as np
import glob, sys, gzip, pickle, os

class Dataset:

    def __init__(self, datafiles):

        self.filenames = glob.glob(datafiles)
        self.load_dataset(self.filenames[0])
        self.ix_file = 0     # Keeps track of which train file we're drawing samples from
        self.samples_consumed = 0  # Num of samples consumed so far.
        self.ix_permutation = np.random.permutation(self.fluxes.shape[0])

    @property
    def fluxes(self):
        return self.data['fluxes']

    @property
    def labels_classifier(self):
        return self.data['labels_classifier']

    @property
    def labels_offset(self):
        return self.data['labels_offset']

    @property
    def col_density(self):
        return self.data['col_density']

    def next_batch(self, batch_size):
        # keep track of how many samples have been consumed and reshuffle & reload after an epoch has elapsed
        if self.samples_consumed > self.fluxes.shape[0]:
            # Load new train data files
            self.ix_file += 1
            self.load_dataset(self.filenames[self.ix_file % len(self.filenames)])
            # Create a new permutation of the data
            self.ix_permutation = np.random.permutation(self.fluxes.shape[0])
            self.samples_consumed = 0

        batch_ix = self.ix_permutation[0:batch_size]
        self.ix_permutation = np.roll(self.ix_permutation, batch_size)
        self.samples_consumed += batch_size

        return self.fluxes[batch_ix, :], \
               self.labels_classifier[batch_ix], \
               self.labels_offset[batch_ix], \
               self.col_density[batch_ix]


    def load_dataset(self, save_file):
        save_file = os.path.splitext(save_file)[0]         # Remove any file extension
        print "Loading data file %s" % save_file
        with gzip.GzipFile(filename=save_file+".pickle", mode='r') as f:
            self.data = pickle.load(f)[0]
        self.data['fluxes'] = np.load(save_file+".npy")
