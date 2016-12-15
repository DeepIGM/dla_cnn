from Id import Id
import os


class Id_GENSAMPLES(Id):
    def __init__(self, ix, hdf5_datafile='../data/gensample_hdf5_files/test_96451_5000.hdf5',
                    json_datafile='../data/gensample_hdf5_files/test_96451_5000.json'):
        super(Id_GENSAMPLES, self).__init__()
        self.ix = ix
        self.hdf5_datafile = hdf5_datafile
        self.json_datafile = json_datafile

    def id_string(self):
        filename = os.path.split(self.hdf5_datafile)[-1]
        basename = os.path.splitext(filename)[0]
        return basename + "_ix_" + "%04d"%self.ix
