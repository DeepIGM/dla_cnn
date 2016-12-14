import scipy.signal as signal


class Prediction(object):
    def __init__(self, peaks_ixs=None, offset_hist=None, offset_conv_sum=None,
                 loc_pred=None, loc_conf=None, offsets=None, density_data=None):
        # Peaks data
        self.peaks_ixs = peaks_ixs
        self.offset_hist = offset_hist
        self.offset_conv_sum = offset_conv_sum

        # Prediction data
        self.loc_pred = loc_pred
        self.loc_conf = loc_conf
        self.offsets = offsets
        self.density_data = density_data

    # Returns a smoothed version of loc_conf
    def smoothed_loc_conf(self, kernel=75):
        # noinspection PyTypeChecker
        return signal.medfilt(self.loc_conf, kernel)

