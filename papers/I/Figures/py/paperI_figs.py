""" Module for the figures of DLA CNN paper
"""

# Imports
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob, os, sys, json
import warnings
import pdb

import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from pkg_resources import resource_filename

from scipy.optimize import minimize
from scipy.stats import chisquare

from astropy import units as u
from astropy.table import Table

from linetools import utils as ltu

from dla_cnn.data_loader import read_sightline
from dla_cnn.data_loader import REST_RANGE
from dla_cnn.data_loader import get_lam_data
from dla_cnn.data_loader import generate_voigt_profile
from dla_cnn.data_loader import get_peaks_for_voigt_scaling
from dla_cnn.data_model.Id_DR7 import Id_DR7


# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
sys.path.append(os.path.abspath("../Vetting/py"))
#from vette_dr7 import load_ml_dr7

default_model = resource_filename('dla_cnn', "models/model_gensample_v7.1")

def fig_varying_confidence(plate=266, fiber=124, fsz=12.):  # Previous Fig 14
    outfile = 'fig_varying_confidence.pdf'
    # Generate ID, load, and process the Sightline
    dr7_id = Id_DR7.from_csv(plate, fiber)
    sightline = read_sightline(dr7_id)
    sightline.process(default_model)
    # Generate model
    loc_conf = sightline.prediction.loc_conf
    peaks_offset = sightline.prediction.peaks_ixs
    offset_conv_sum = sightline.prediction.offset_conv_sum
    # smoothed_sample = sightline.prediction.smoothed_loc_conf()

    PLOT_LEFT_BUFFER = 50       # The number of pixels to plot left of the predicted sightline
    dlas_counter = 0

    #pp = PdfPages(filename)

    full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    lam_rest = full_lam_rest[full_ix_dla_range]
    lam = full_lam[full_ix_dla_range]

    y = sightline.flux
    y_plot_range = np.mean(y[y > 0]) * 3.0
    #xlim = [REST_RANGE[0]-PLOT_LEFT_BUFFER, lam_rest[-1]]
    xlim = (3800., 4800.)
    ylim = [-2, y_plot_range]

    n_dlas = len(sightline.prediction.peaks_ixs)

    # Plot DLA range
    n_rows = 2 + (1 if n_dlas>0 else 0) #+ n_dlas
    fig = plt.figure(figsize=(5, 9))
    gs = gridspec.GridSpec(3,2)
    #axtxt = fig.add_subplot(n_rows,1,1)
    #axsl = fig.add_subplot(n_rows,1,2)
    axsl = plt.subplot(gs[0,:])
    #axloc = fig.add_subplot(n_rows,1,3)
    axloc = plt.subplot(gs[1,:])

    #axsl.set_xlabel("Rest frame sightline in region of interest for DLAs with z_qso = [%0.4f]" % sightline.z_qso)
    axsl.set_ylabel(r'Flux ($10^{-17} \rm erg \, s^{-1} \, cm^{-2} \, A^{-1}$)')
    axsl.set_xlabel('Wavelength (Ang)')
    axsl.set_ylim(ylim)
    axsl.set_xlim(xlim)
    axsl.plot(full_lam, sightline.flux, '-k')
    set_fontsize(axsl,fsz)

    # Plot 0-line
    axsl.plot(xlim, [0.]*2, 'g--')

    # Plot z_qso line over sightline
    # axsl.plot((1216, 1216), (ylim[0], ylim[1]), 'k-', linewidth=2, color='grey', alpha=0.4)

    '''
    # Plot observer frame ticks
    axupper = axsl.twiny()
    axupper.set_xlim(xlim)
    xticks = np.array(axsl.get_xticks())[1:-1]
    axupper.set_xticks(xticks)
    axupper.set_xticklabels((xticks * (1 + sightline.z_qso)).astype(np.int32))
    '''

    # Plot given DLA markers over location plot
    for dla in sightline.dlas if sightline.dlas is not None else []:
        dla_rest = dla.central_wavelength / (1+sightline.z_qso)
        axsl.plot((dla_rest, dla_rest), (ylim[0], ylim[1]), 'g--')

    # Plot localization
    #axloc.set_xlabel("DLA Localization confidence & localization prediction(s)")
    #axloc.set_ylabel("Identification")
    axloc.plot(lam, loc_conf, color='deepskyblue')
    axloc.set_ylim([0, 1])
    axloc.set_xlim(xlim)

    # Classification results
    #textresult = "Classified %s (%0.5f ra / %0.5f dec) with %d DLAs/sub dlas/Ly-B\n" \
    #             % (sightline.id.id_string(), sightline.id.ra, sightline.id.dec, n_dlas)

    # Plot localization histogram
    pkclr = 'blue'
    #axloc.scatter(lam, sightline.prediction.offset_hist, s=6, color='orange')
    axloc.plot(lam, sightline.prediction.offset_conv_sum, color='green')
    #axloc.plot(lam, sightline.prediction.smoothed_conv_sum(), color='yellow', linestyle='-', linewidth=0.25)

    axloc.set_ylabel(r'Flux ($10^{-17} \rm erg \, s^{-1} \, cm^{-2} \, A^{-1}$)')
    axloc.set_xlabel('Wavelength (Ang)')
    set_fontsize(axloc,fsz)

    # Plot '+' peak markers
    if len(peaks_offset) > 0:
        axloc.plot(lam[peaks_offset],
                   np.minimum(1, offset_conv_sum[peaks_offset]), '+', mew=5, ms=10, color='green', alpha=1)

    #
    # For loop over each DLA identified
    #
    for dlaix, peak in zip(range(0,n_dlas), peaks_offset):
        # Some calculations that will be used multiple times
        dla_z = lam_rest[peak] * (1 + sightline.z_qso) / 1215.67 - 1

        # Sightline plot transparent marker boxes
        axsl.fill_between(lam[peak - 10:peak + 10], y_plot_range, -2, color='green', lw=0, alpha=0.1)
        axsl.fill_between(lam[peak - 30:peak + 30], y_plot_range, -2, color='green', lw=0, alpha=0.1)
        axsl.fill_between(lam[peak - 50:peak + 50], y_plot_range, -2, color='green', lw=0, alpha=0.1)
        axsl.fill_between(lam[peak - 70:peak + 70], y_plot_range, -2, color='green', lw=0, alpha=0.1)

        density_pred_per_this_dla, mean_col_density_prediction, std_col_density_prediction, bias_correction = \
            sightline.prediction.get_coldensity_for_peak(peak)
        '''
        # Plot column density measures with bar plots
        # density_pred_per_this_dla = sightline.prediction.density_data[peak-40:peak+40]
        dlas_counter += 1
        # mean_col_density_prediction = float(np.mean(density_pred_per_this_dla))

        pltix = fig.add_subplot(n_rows, 1, 5+dlaix)
        pltix.bar(np.arange(0, density_pred_per_this_dla.shape[0]), density_pred_per_this_dla, 0.25)
        pltix.set_xlabel("Individual Column Density estimates for peak @ %0.0fA, +/- 0.3 of mean. Bias adjustment of %0.3f added. " %
                         (lam_rest[peak], float(bias_correction)) +
                         "Mean: %0.3f - Median: %0.3f - Stddev: %0.3f" %
                         (mean_col_density_prediction, float(np.median(density_pred_per_this_dla)),
                          float(std_col_density_prediction)))
        pltix.set_ylim([mean_col_density_prediction - 0.3, mean_col_density_prediction + 0.3])
        pltix.plot(np.arange(0, density_pred_per_this_dla.shape[0]),
                   np.ones((density_pred_per_this_dla.shape[0]), np.float32) * mean_col_density_prediction)
        pltix.set_ylabel("Column Density")

        # Add DLA to test result
        absorber_type = "Ly-b" if sightline.is_lyb(peak) else "DLA" if mean_col_density_prediction >= 20.3 else "sub dla"
        dla_text = \
            "%s at: %0.0fA rest / %0.0fA observed / %0.4f z, w/ confidence %0.2f, has Column Density: %0.3f" \
            % (absorber_type,
               lam_rest[peak],
               lam_rest[peak] * (1 + sightline.z_qso),
               dla_z,
               min(1.0, float(sightline.prediction.offset_conv_sum[peak])),
               mean_col_density_prediction)
        textresult += " > " + dla_text + "\n"
        '''

        #
        # Plot DLA zoom view with voigt overlay
        #
        # Generate the voigt model using astropy, linetools, etc.
        voigt_flux, voigt_wave = generate_voigt_profile(dla_z, mean_col_density_prediction, full_lam)
        # get peaks
        ixs_mypeaks = get_peaks_for_voigt_scaling(sightline, voigt_flux)
        # get indexes where voigt profile is between 0.2 and 0.95
        observed_values = sightline.flux[ixs_mypeaks]
        expected_values = voigt_flux[ixs_mypeaks]
        # Minimize scale variable using chi square measure
        opt = minimize(lambda scale: chisquare(observed_values, expected_values * scale)[0], 1)
        opt_scale = opt.x[0]

        dla_min_text = \
            "%0.0fA rest / %0.0fA observed - NHI %0.3f" \
            % (lam_rest[peak],
               lam_rest[peak] * (1 + sightline.z_qso),
               mean_col_density_prediction)

        inax = plt.subplot(gs[2,dlaix])
        inax.plot(full_lam, sightline.flux, '-k', lw=1.2)
        #inax.plot(full_lam[ixs_mypeaks], sightline.flux[ixs_mypeaks], '+',
        #          mew=5, ms=10, color='orange', alpha=1)
        inax.plot(voigt_wave, voigt_flux * opt_scale, 'r--', lw=3.0)
        inax.set_ylim(ylim)
        # convert peak to index into full_lam range for plotting
        peak_full_lam = np.nonzero(np.cumsum(full_ix_dla_range) > peak)[0][0]
        inax.set_xlim([full_lam[peak_full_lam-150],full_lam[peak_full_lam+150]])
        inax.axhline(0, color='grey')
        set_fontsize(inax,fsz-2.)

        #
        # Plot legend on location graph
        #
        axloc.legend(['DLA classifier', 'Localization', 'DLA peak', 'Localization histogram'],
                     bbox_to_anchor=(1.0, 1.05), loc='upper right')

    '''
    # Display text
    axtxt.text(0, 0, textresult, family='monospace', fontsize='xx-large')
    axtxt.get_xaxis().set_visible(False)
    axtxt.get_yaxis().set_visible(False)
    axtxt.set_frame_on(False)
    '''

    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    print("Wrote: {:s}".format(outfile))
    plt.close('all')


def fig_n07_no_detect(ml_dlasurvey=None):
    outfil='fig_n07_no_detect.pdf'
    # Load PN
    pn_dr7_file = '../Analysis/noterdaeme_dr7.fits'
    pn_dr7 = Table.read(pn_dr7_file)
    # Load vette file
    vette_file = '../Vetting/vette_dr7_pn.json'
    vdr7 = ltu.loadjson(vette_file)
    in_ml = np.array(vdr7['in_ml'])
    pn_ml_idx = np.array(vdr7['pn_idx'])
    not_in_pn = np.array(vdr7['not_in_pn'])

    # Missed DLAs
    misses = pn_ml_idx == -1

    # Start the plot
    plt.figure(figsize=(8, 4))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # dz
    ax1 = plt.subplot(gs[0])
    zabs = pn_dr7['zabs'][misses]
    ax1.hist(zabs)
    # Axes
    #ax1.set_yscale("log", nonposy='clip')
    #ax1.set_ylim(1., 3000.)
    #ax1.set_xlim(-0.02, 0.02)
    #ax1.xaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax1.set_xlabel(r'$z_{\rm miss}$')
    ax1.set_ylabel('N')

    NHI = pn_dr7['logN_HI_'][misses]
    ax2 = plt.subplot(gs[1])
    ax2.hist(NHI)
    # Axes
    #ax1.set_yscale("log", nonposy='clip')
    #ax1.set_ylim(1., 3000.)
    #ax2.set_xlim(-0.02, 0.02)
    #ax2.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax2.set_xlabel(r'$\log \, N_{\rm HI, miss}$')
    ax2.set_ylabel('N')

    # Legend
    #legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
    #                  handletextpad=0.3, fontsize='medium', numpoints=1)
    set_fontsize(ax1,15.)
    set_fontsize(ax2,15.)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfil)
    plt.close()
    print("Wrote {:s}".format(outfil))


def fig_not_in_n07(ml_dlasurvey=None):
    """ DLAs in the ML but not in N07 (and mainly not in DR5 either)
    """
    outfil='fig_not_in_n07.pdf'
    if ml_dlasurvey is None:
        _, ml_dlasurvey = load_ml_dr7()
    # Load PN
    pn_dr7_file = '../Analysis/noterdaeme_dr7.fits'
    pn_dr7 = Table.read(pn_dr7_file)
    # Load vette file
    vette_file = '../Vetting/vette_dr7_pn.json'
    vdr7 = ltu.loadjson(vette_file)
    in_ml = np.array(vdr7['in_ml'])
    pn_ml_idx = np.array(vdr7['pn_idx'])
    not_in_pn = np.array(vdr7['not_in_pn'])
    print("There are {:d} DLAs in ML and not in N07".format(len(not_in_pn)))

    # Start the plot
    plt.figure(figsize=(8, 4))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # dz
    ax1 = plt.subplot(gs[0])
    zabs = ml_dlasurvey.zabs[not_in_pn]
    ax1.hist(zabs)
    # Axes
    #ax1.set_yscale("log", nonposy='clip')
    #ax1.set_ylim(1., 3000.)
    #ax1.set_xlim(-0.02, 0.02)
    #ax1.xaxis.set_major_locator(plt.MultipleLocator(0.01))
    #ax1.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax1.set_xlabel(r'$z_{\rm not \, in}$')
    ax1.set_ylabel('N')

    NHI = ml_dlasurvey.NHI[not_in_pn]
    ax2 = plt.subplot(gs[1])
    ax2.hist(NHI)
    # Axes
    #ax1.set_yscale("log", nonposy='clip')
    #ax1.set_ylim(1., 3000.)
    #ax2.set_xlim(-0.02, 0.02)
    #ax2.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax2.set_xlabel(r'$\log \, N_{\rm HI, not \, in}$')
    ax2.set_ylabel('N')

    # Legend
    #legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
    #                  handletextpad=0.3, fontsize='medium', numpoints=1)
    set_fontsize(ax1,15.)
    set_fontsize(ax2,15.)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfil)
    plt.close()
    print("Wrote {:s}".format(outfil))


def fig_n07_vs_ml(ml_dlasurvey=None):
    """ Plot Dz and DNHI for overlapping DLAs in PN vs. ML
    """
    outfil='fig_n07_vs_ml.pdf'
    # Load DLA samples
    if ml_dlasurvey is None:
        _, ml_dlasurvey = load_ml_dr7()
    # Load PN
    pn_dr7_file = '../Analysis/noterdaeme_dr7.fits'
    pn_dr7 = Table.read(pn_dr7_file)
    # Load vette file
    vette_file = '../Vetting/vette_dr7_pn.json'
    vdr7 = ltu.loadjson(vette_file)
    in_ml = np.array(vdr7['in_ml'])
    pn_ml_idx = np.array(vdr7['pn_idx'])
    not_in_pn  = np.array(vdr7['not_in_pn'])
    cut_pn = pn_dr7[in_ml]

    # Start the plot
    plt.figure(figsize=(8, 4))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # dz
    ax1 = plt.subplot(gs[0])
    gdm = pn_ml_idx >= 0
    iLLS = pn_ml_idx == -9
    print('There are {:d} DLAs N07'.format(np.sum(pn_dr7['logN_HI_'] >= 20.299)))
    print('There are {:d} DLAs that matched to ML DLAs'.format(np.sum(gdm)))
    print('There are {:d} DLAs that matched to ML LLS'.format(np.sum(iLLS)))
    dz = cut_pn['zabs'][gdm]-ml_dlasurvey.zabs[pn_ml_idx[gdm]]
    print('median dz = {}, std dz = {}'.format(np.median(dz), np.std(dz)))
    ax1.hist(dz)
    # Axes
    #ax1.set_yscale("log", nonposy='clip')
    #ax1.set_ylim(1., 3000.)
    ax1.set_xlim(-0.02, 0.02)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax1.set_xlabel(r'$\Delta \, z$')
    ax1.set_ylabel('N')

    dNHI = cut_pn['logN_HI_'][gdm]-ml_dlasurvey.NHI[pn_ml_idx[gdm]]
    print('median dNHI = {}, std dNHI = {}'.format(np.median(dNHI), np.std(dNHI)))
    ax2 = plt.subplot(gs[1])
    ax2.hist(dNHI)
    # Axes
    #ax1.set_yscale("log", nonposy='clip')
    #ax1.set_ylim(1., 3000.)
    #ax2.set_xlim(-0.02, 0.02)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax2.set_xlabel(r'$\Delta \, \log \, N_{\rm HI}$')
    ax2.set_ylabel('N')

    # Legend
    #legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
    #                  handletextpad=0.3, fontsize='medium', numpoints=1)
    set_fontsize(ax1,15.)
    set_fontsize(ax2,15.)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfil)
    plt.close()
    print("Wrote {:s}".format(outfil))


def set_fontsize(ax,fsz):
    '''
    Generate a Table of columns and so on
    Restrict to those systems where flg_clm > 0

    Parameters
    ----------
    ax : Matplotlib ax class
    fsz : float
      Font size
    '''
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsz)


#### ########################## #########################
def main(flg_fig):

    if flg_fig == 'all':
        flg_fig = np.sum( np.array( [2**ii for ii in range(5)] ))
    else:
        flg_fig = int(flg_fig)

    # Compare dz and dNHI between N07 and ML
    if flg_fig & (2**0):
        fig_n07_vs_ml()

    # Plot missed DLAs from N07
    if flg_fig & (2**1):
        fig_n07_no_detect()

    # Plot missed DLAs from N07
    if flg_fig & (2**2):
        fig_varying_confidence()



# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2**0   # dz, dNHI from N07 to ML
        #flg_fig += 2**1   # Missed DLAs in N07
        flg_fig += 2**2   # Two DLAs with differing confidence
        #flg_fig += 2**3   # ne/nH
        #flg_fig += 2**4   # Average DM
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
