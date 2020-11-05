import numpy as np
import glob, sys, gzip, pickle, os, multiprocessing.dummy

class Dataset:

    def __init__(self, datafiles, BUFFERSIZE=3000000):
        self.p = multiprocessing.dummy.Pool(1)
        self.future = None

        self.data = None
        self.filenames = glob.glob(datafiles)#datafiles
        self.sample_count = 0
        assert len(self.filenames) > 0
        for ff in self.filenames:
            r= np.load(ff,allow_pickle = True,encoding='latin1').item()
            for f in r.keys():
                n_samples = len(r[f]['labels_classifier']) #npt文件的问题：存储之后不是纯粹的数组，怎么读取出来？
                self.sample_count += n_samples
                self.kernel_size = r[f]['FLUX'].shape[1]
                print("DEBUG> init dataset file loop, counting samples in [%s]: %d" % (f,n_samples))

        self.BUFFERSIZE = min(BUFFERSIZE, self.sample_count)

        # for reading in from multiple files
        self.samples_consumed = 0
        self.batch_consumed = 0

        self.ix_permutation = np.random.permutation(self.sample_count)
        self.get_next_buffer()
        self.batch_range = np.random.permutation(self.BUFFERSIZE)       # used to iterate through the already permuted buffer

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
        self.batch_range = np.roll(self.batch_range, batch_size*-1)
        self.batch_consumed += batch_size

        if self.batch_consumed >= self.BUFFERSIZE:
            # self.ix_permutation = np.random.permutation(self.sample_count)
            self.batch_consumed = 0
            self.batch_range = np.random.permutation(self.BUFFERSIZE)
            self.get_next_buffer()

        return self.fluxes[batch_ix, :], \
               self.labels_classifier[batch_ix], \
               self.labels_offset[batch_ix], \
               self.col_density[batch_ix]


    # caches 1 buffer ahead loaded asynchronusly, returns the cacheed buffer and starts a new one loading
    def get_next_buffer(self):
        print("DEBUG> get_next_buffer called %i files in set" % len(self.filenames))
        if not self.data:
            self.future = self.p.apply_async(self.load_dataset_slice)

        self.data = self.future.get()
        # print np.histogram(self.data['col_density']) #debug statement
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
            self.ix_permutation = np.roll(self.ix_permutation, self.BUFFERSIZE*-1)

        distributed_ix = 0          # Counts samples across all files
        buffer_count = 0            # Pointer to location in buffer
        print ("DEBUG> Enter file load loop")
        for f in self.filenames:
            x = np.load(f,allow_pickle = True,encoding='latin1').item()
            for kk in x.keys():

                x_len = x[kk]['labels_classifier'].shape[0]
                x_ixs = samples_ix[(samples_ix >= distributed_ix) & (samples_ix < distributed_ix + x_len)] - distributed_ix
                distributed_ix += x_len
            # print "DEBUG> FILE LOOP: [%s] loaded %d samples" % (f, len(x_ixs))

                data['fluxes'][buffer_count:buffer_count + len(x_ixs)] = x[kk]['FLUX'][x_ixs]
                data['labels_classifier'][buffer_count:buffer_count + len(x_ixs)] = x[kk]['labels_classifier'][x_ixs]
                data['labels_offset'][buffer_count:buffer_count + len(x_ixs)] = x[kk]['labels_offset'][x_ixs]
                data['col_density'][buffer_count:buffer_count + len(x_ixs)] = x[kk]['col_density'][x_ixs]

                buffer_count += len(x_ixs)
        print ("DEBUG> File load loop complete")

        return data
