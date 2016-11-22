import Id

class Id_GENSAMPLES(Id.Id):
    def __init__(self, ix, hdf5_datafile='../data/training_100.hdf5',
                    json_datafile='../data/training_100.json'):
        self.ix = ix
        self.hdf5_datafile = hdf5_datafile
        self.json_datafile = json_datafile