import scipy.signal as signal
import numpy as np


class Prediction(object):
    def __init__(self, peaks_ixs=None, offset_hist=None, offset_conv_sum=None,
                 loc_pred=None, loc_conf=None, offsets=None, density_data=None):
        # Peaks data
        # self.peaks_ixs = np.sort(peaks_ixs) if peaks_ixs is not None else None
        self._peaks_ixs = None
        self.peaks_ixs = peaks_ixs
        self.offset_hist = offset_hist
        self.offset_conv_sum = offset_conv_sum

        # Prediction data
        self.loc_pred = loc_pred
        self.loc_conf = loc_conf
        self.offsets = offsets
        self.density_data = density_data
#
    @property
    def peaks_ixs(self):
        return self._peaks_ixs

    @peaks_ixs.setter
    def peaks_ixs(self, peaks_ixs):
        self._peaks_ixs = np.sort(peaks_ixs) if peaks_ixs is not None else None

    # Returns a smoothed version of loc_conf
    def smoothed_loc_conf(self, kernel=75):
        # noinspection PyTypeChecker
        return signal.medfilt(self.loc_conf, kernel)


    def smoothed_conv_sum(self, kernel=9):
        return signal.medfilt(self.offset_conv_sum, kernel)


    # Returns the column density estimates for a specific peak and the mean
    # Handles cases where the column density is too close to another DLA
    # Takes a bias adjustment polynomial to adjust the column density, returns the adjustment factor
    # Note, the bias adjustment polynomial is hard coded here, but it would more logically be stored with the model, this is time-saving shortcut for now.
    def get_coldensity_for_peak(self, peak_ix,
                                bias_adjust=None):#(1.068052674832284720807251687802, -1.437592819741396965582680422813)):
        normal_range = 30

        is_close_dla_left = np.any((self.peaks_ixs < peak_ix) & (self.peaks_ixs >= peak_ix-normal_range*2))
        is_close_dla_right = np.any((self.peaks_ixs > peak_ix) & (self.peaks_ixs <= peak_ix+normal_range*2))
        if is_close_dla_left and is_close_dla_right:
            # Special case where a DLA is pinned between two close DLAs
            range_left = (peak_ix - max(self.peaks_ixs[self.peaks_ixs<peak_ix]))/2
            range_right = (min(self.peaks_ixs[self.peaks_ixs>peak_ix]) - peak_ix)/2
            col_densities = self.density_data[peak_ix - range_left:peak_ix + range_right]
        else:
            # Take the left side predictions or right side predictions or both
            range_left = 0 if is_close_dla_left else normal_range
            range_right = 0 if is_close_dla_right else normal_range
            col_densities = self.density_data[peak_ix - range_left:peak_ix + range_right]

        mean_col_density = np.mean(col_densities)
        bias_correction = np.polyval(bias_adjust, mean_col_density) - mean_col_density if bias_adjust else 0.0

        return col_densities + bias_correction, \
               mean_col_density + bias_correction, \
               np.std(col_densities), \
               bias_correction
