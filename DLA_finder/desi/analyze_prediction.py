import numpy as np 
from dla_cnn.data_model.Prediction import Prediction
from dla_cnn.spectra_utils import get_lam_data
import scipy.signal as signal
def compute_peaks(sightline,PEAK_THRESH):
    # Threshold to accept a peak0.2
    PEAK_SEPARATION_THRESH = 0.1        # Peaks must be separated by a valley at least this low

    # Translate relative offsets to histogram
    offset_to_ix = np.arange(len(sightline.prediction.offsets)) + sightline.prediction.offsets
    offset_to_ix[offset_to_ix < 0] = 0
    offset_to_ix[offset_to_ix >= len(sightline.prediction.offsets)] = len(sightline.prediction.offsets)
    offset_hist, ignore_offset_range = np.histogram(offset_to_ix, bins=np.arange(0,len(sightline.prediction.offsets)+1))

    # Somewhat arbitrary normalization
    offset_hist = offset_hist / 80.0

    po = np.pad(offset_hist, 2, 'constant', constant_values=np.mean(offset_hist))
    offset_conv_sum = (po[:-4] + po[1:-3] + po[2:-2] + po[3:-1] + po[4:])
    smooth_conv_sum = signal.medfilt(offset_conv_sum, 9)
    # ensures a 0 value at the beginning and end exists to avoid an unnecessarily pathalogical case below
    smooth_conv_sum[0] = 0
    smooth_conv_sum[-1] = 0

    peaks_ixs = []
    while True:
        peak = np.argmax(smooth_conv_sum)   # Returns the first occurace of the max
        # exit if we're no longer finding valid peaks
        if smooth_conv_sum[peak] < PEAK_THRESH:
            break
        # skip this peak if it's off the end or beginning of the sightline
        if peak <= 10 :#or peak >= REST_RANGE[2]-10:
            smooth_conv_sum[max(0,peak-15):peak+15] = 0
            continue
        # move to the middle of the peak if there are multiple equal values
        ridge = 1
        while smooth_conv_sum[peak] == smooth_conv_sum[peak+ridge]:
            ridge += 1
        peak = peak + ridge//2
        peaks_ixs.append(peak)

        # clear points around the peak, that is, anything above PEAK_THRESH in order for a new DLA to be identified the peak has to dip below PEAK_THRESH
        clear_left = smooth_conv_sum[0:peak+1] < PEAK_SEPARATION_THRESH # something like: 1 0 0 1 1 1 0 0 0 0
        clear_left = np.nonzero(clear_left)[0][-1]+1                    # Take the last value and increment 1
        clear_right = smooth_conv_sum[peak:] < PEAK_SEPARATION_THRESH   # something like 0 0 0 0 1 1 1 0 0 1
        clear_right = np.nonzero(clear_right)[0][0]+peak                # Take the first value & add the peak offset
        smooth_conv_sum[clear_left:clear_right] = 0

    sightline.prediction.peaks_ixs = peaks_ixs
    sightline.prediction.offset_hist = offset_hist
    sightline.prediction.offset_conv_sum = offset_conv_sum
    #if peaks_ixs==[]:
        #print(sightline.id,np.amax(smooth_conv_sum))
    return sightline

def analyze_pred(sightline,pred,conf, offset, coldensity,PEAK_THRESH):
    for i in range(0,len(pred)):#exclude offset when pred=0
        if (pred[i]==0):
            offset[i]=0
    sightline.prediction = Prediction(loc_pred=pred, loc_conf=conf, offsets=offset, density_data=coldensity)
    # get prediction for each sightline
    compute_peaks(sightline,PEAK_THRESH)
    sightline.prediction.smoothed_loc_conf()
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso)
    kernelrangepx = 200
    cut=((np.nonzero(ix_dla_range)[0])>=kernelrangepx)&((np.nonzero(ix_dla_range)[0])<=(len(lam)-kernelrangepx-1)) 
    #get input lam array
    lam_analyse=lam[ix_dla_range][cut]
    dla_sub_lyb=[]
    for peak in sightline.prediction.peaks_ixs:
        peak_lam_rest=lam_rest[ix_dla_range][cut][peak]
        peak_lam_spectrum = lam_analyse[peak]
        z_dla = float(peak_lam_spectrum) / 1215.67 - 1
        _, mean_col_density_prediction, std_col_density_prediction, bias_correction =             sightline.prediction.get_coldensity_for_peak(peak)

        absorber_type =  "DLA" if mean_col_density_prediction >= 20.3 else "LYB" if sightline.is_lyb(peak) else "SUBDLA"
        
        abs_dict =  {
            'rest': float(peak_lam_rest),
            'spectrum': float(peak_lam_spectrum),
            'z_dla':float(z_dla),
            'dla_confidence': min(1.0,float(sightline.prediction.offset_conv_sum[peak])),
            'column_density': float(mean_col_density_prediction),
            'std_column_density': float(std_col_density_prediction),
            'column_density_bias_adjust': float(bias_correction),
            'type': absorber_type
        }
        dla_sub_lyb.append(abs_dict)
    return dla_sub_lyb




