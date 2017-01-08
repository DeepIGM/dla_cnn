import numpy as np
import glob, sys, gzip, pickle, os, multiprocessing.dummy

class Dataset:

    def __init__(self, datafiles, BUFFERSIZE=2000000):
        self.p = multiprocessing.dummy.Pool(1)
        self.future = None

        self.data = None
        self.filenames = glob.glob(datafiles)
        self.sample_count = 0
        for f in self.filenames:
            print "DEBUG> init dataset file loop"
            r = np.load(f)
            self.sample_count += len(r['labels_classifier'])
            self.kernel_size = r['flux'].shape[1]

        self.BUFFERSIZE = min(BUFFERSIZE, self.sample_count)

        # for reading in from multiple files
        self.samples_consumed = 0

        self.ix_permutation = np.random.permutation(self.sample_count)
        self.get_next_buffer()
        self.batch_range = np.arange(self.BUFFERSIZE)       # used to iterate through the buffer

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
        batch_ix = self.batch_range[0:batch_size]
        self.batch_range = np.roll(self.batch_range, batch_size)

        if self.batch_range[0] > self.batch_range[batch_size]:
            self.ix_permutation = np.random.permutation(self.sample_count)
            self.get_next_buffer()

        return self.fluxes[batch_ix, :], \
               self.labels_classifier[batch_ix], \
               self.labels_offset[batch_ix], \
               self.col_density[batch_ix]


    # caches 1 buffer ahead loaded asynchronusly, returns the cacheed buffer and starts a new one loading
    def get_next_buffer(self):
        print "DEBUG> get_next_buffer called", len(self.filenames), self.filenames[0]
        if self.sample_count == self.BUFFERSIZE:
            if not self.data:
                self.data = self.load_dataset_slice()
            return

        # An extra run the first time to initialize things
        if not self.future:
            self.future = self.p.apply_async(self.load_dataset_slice)

        self.data = self.future.get()
        self.future = self.p.apply_async(self.load_dataset_slice)


    # Loads a set of randomly selected samples from across all files and returns the data
    def load_dataset_slice(self):
        data = {}
        data['fluxes'] = np.empty((self.BUFFERSIZE, self.kernel_size), dtype=np.float32)
        data['labels_classifier'] = np.empty((self.BUFFERSIZE), dtype=np.float32)
        data['labels_offset'] = np.empty((self.BUFFERSIZE), dtype=np.float32)
        data['col_density'] = np.empty((self.BUFFERSIZE), dtype=np.float32)

        samples_ix = self.ix_permutation[0:self.BUFFERSIZE]
        self.samples_consumed += self.BUFFERSIZE
        if self.samples_consumed >= self.sample_count:                     # Reshuffle or roll
            self.samples_consumed = 0  # Num of samples consumed so far.
            self.ix_permutation = np.random.permutation(self.sample_count)
        else:
            self.ix_permutation = np.roll(self.ix_permutation, self.BUFFERSIZE)

        distributed_ix = 0          # Counts samples across all files
        buffer_count = 0            # Pointer to location in buffer
        for f in self.filenames:
            print "DEBUG> FILE LOOP"
            x = np.load(f)
            x_len = x['labels_classifier'].shape[0]
            x_ixs = samples_ix[(samples_ix >= distributed_ix) & (samples_ix < distributed_ix + x_len)] - distributed_ix
            distributed_ix += x_len

            data['fluxes'][buffer_count:buffer_count + len(x_ixs)] = x['flux'][x_ixs]
            data['labels_classifier'][buffer_count:buffer_count + len(x_ixs)] = x['labels_classifier'][x_ixs]
            data['labels_offset'][buffer_count:buffer_count + len(x_ixs)] = x['labels_offset'][x_ixs]
            data['col_density'][buffer_count:buffer_count + len(x_ixs)] = x['col_density'][x_ixs]

            buffer_count += len(x_ixs)

        return data