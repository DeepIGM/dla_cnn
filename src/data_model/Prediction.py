
class Prediction(object):
    def __init__(self, peaks_data=None, loc_pred=None, loc_conf=None, offsets=None, density_data=None):
        self.peaks_data = peaks_data
        self.loc_pred = loc_pred
        self.loc_conf = loc_conf
        self.offsets = offsets
        self.density_data = density_data
