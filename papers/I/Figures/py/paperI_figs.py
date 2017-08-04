""" Module for the figures of DLA CNN paper
"""

# Imports
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob, os, sys
import warnings
import pdb

import matplotlib as mpl
import h5py

mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from pkg_resources import resource_filename

from scipy.optimize import minimize
from scipy.stats import chisquare

from astropy import units as u
from astropy.table import Table

from linetools import utils as ltu
from linetools.spectra import io as lsio

from specdb.specdb import IgmSpec

from pyigm.surveys.dlasurvey import DLASurvey, dla_stat

CNN_result_path = '/home/xavier/Projects/ML_DLA_results/CNN/'

if sys.version[0] == '3':
    pass
else: # Only Python 2.7
    from dla_cnn.data_loader import REST_RANGE
    from dla_cnn.data_loader import read_sightline
    from dla_cnn.data_loader import get_lam_data
    from dla_cnn.data_model.Id_DR7 import Id_DR7
    from dla_cnn.data_model.Id_DR12 import Id_DR12
    from dla_cnn.absorption import generate_voigt_model
from dla_cnn.io import load_ml_dr7
from dla_cnn import training_set as tset


# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#sys.path.append(os.path.abspath("../Vetting/py"))
#from vette_dr7 import load_ml_dr7

default_model = resource_filename('dla_cnn', "models/model_gensample_v7.1")

def init_for_ipython():
    #from imp import reload
    #import paperI_figs as pfigs
    _, ml_dlasurvey = load_ml_dr7()
    return ml_dlasurvey

def fig_ignore_flux(fsz=12.):  # Previous Fig 13
    #jfrom dla_cnn.data_loader import generate_voigt_model
    from dla_cnn.absorption import generate_voigt_model
    plates=(2111,2111)
    fibers=(525,525)
    xlims = ((4640, 4950), (4640, 4950))
    ylims = ((-1., 8), (-1, 8.))

    # Plot
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2,1)

    for jj, plate,fiber,xlim,ylim in zip(range(len(plates)), plates, fibers, xlims, ylims):
        # Process sightline
        dr7_id = Id_DR7.from_csv(plate, fiber)
        sightline = read_sightline(dr7_id)
        sightline.process(default_model)

        full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)

        # Plot
        ax = plt.subplot(gs[jj])
        ax.plot(full_lam, sightline.flux, 'k', drawstyle='steps-mid')
        ax.plot(xlim, [0.]*2, '--', color='gray')

        ax.set_ylabel(r'F$_\lambda$ ($10^{-17} \rm erg \, s^{-1} \, cm^{-2} \, A^{-1}$)')
        ax.set_xlabel('Wavelength (Ang)')
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

        # DLA
        assert len(sightline.dlas) == 1
        voigt_wave, voigt_model, _ = generate_voigt_model(sightline, sightline.dlas[0])
        ax.plot(voigt_wave, voigt_model, 'r--')
        set_fontsize(ax,fsz)
    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    outfile='fig_ignore_flux.pdf'
    plt.savefig(outfile)
    print("Wrote: {:s}".format(outfile))
    plt.close('all')

def fig_labels(plate=4484, fiber=364):

    # Generate ID, load, and process the Sightline
    dr12_id = Id_DR12.from_csv(plate, fiber)
    sightline = read_sightline(dr12_id)
    sightline.process(default_model)
    # Generate model
    loc_conf = sightline.prediction.loc_conf
    #peaks_offset = sightline.prediction.peaks_ixs
    #offset_conv_sum = sightline.prediction.offset_conv_sum
    offsets = sightline.prediction.offsets
    NHI = sightline.prediction.density_data
    full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(sightline.loglam,
                                                              sightline.z_qso, REST_RANGE)
    gdi = np.where(full_ix_dla_range)[0]
    # Cut down
    cut_idx = np.arange(300,len(gdi))
    gdi = gdi[cut_idx]

    # Start the plot
    fig = plt.figure(figsize=(6, 8))
    plt.clf()
    gs = gridspec.GridSpec(4,1)
    #xlim = (3600., 4450)
    lsz = 13.

    # Flux
    axsl = plt.subplot(gs[0])
    axsl.plot(full_lam[gdi], sightline.flux[gdi], '-k', lw=0.5)
    set_fontsize(axsl, lsz)
    axsl.set_ylabel('Relative Flux')
    axsl.get_xaxis().set_ticks([])
    ylim = (-0.5, 8.)
    axsl.set_ylim(ylim)

    # Confidence
    axc = plt.subplot(gs[1])
    axc.plot(full_lam[gdi], loc_conf[cut_idx], color='blue')
    set_fontsize(axc, lsz)
    axc.set_ylabel('Classification')
    axc.get_xaxis().set_ticks([])

    # Offset
    axo = plt.subplot(gs[2])
    axo.plot(full_lam[gdi], offsets[cut_idx], color='green')
    set_fontsize(axo, lsz)
    axo.set_ylabel('Localization (pix)')
    axo.get_xaxis().set_ticks([])

    # NHI
    axN = plt.subplot(gs[3])
    axN.plot(full_lam[gdi], NHI[cut_idx], color='red')
    set_fontsize(axN, lsz)
    axN.set_ylabel(r'$\log \, N_{\rm HI}$')
    axN.set_xlabel('Wavelength')

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    outfile = 'fig_labels.pdf'
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))


def fig_dla_confidence(plate=4484, fiber=364):
    """ Plot the confidence values for 2 DLAs
    """
    # Generate ID, load, and process the Sightline
    dr12_id = Id_DR12.from_csv(plate, fiber)
    sightline = read_sightline(dr12_id)
    sightline.process(default_model)
    # Generate model
    full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(sightline.loglam,
                                                              sightline.z_qso, REST_RANGE)

    # Start the plot
    fig = plt.figure(figsize=(6, 6))
    plt.clf()
    gs = gridspec.GridSpec(2,2)
    lsz = 13.

    # Loop on DLAs
    # DLA 1

    for ss in range(2):
        if ss == 0:
            wv1 = (3925., 4025.)
            ylim = (-0.9, 5.2)
        else:
            wv1 = (4100., 4240.)
        iwv = (full_lam > wv1[0]) & (full_lam < wv1[1])
        ixwv = (full_lam[full_ix_dla_range] > wv1[0]) & (full_lam[full_ix_dla_range] < wv1[1])
        gdi = np.where(full_ix_dla_range & iwv)[0]
        gdix = np.where(ixwv)[0]

        # Flux
        ax1 = plt.subplot(gs[0,ss])
        ax1.plot(full_lam[gdi], sightline.flux[gdi], '-k', lw=1.3, drawstyle='steps-mid')
        set_fontsize(ax1, lsz)
        ax1.set_ylabel('Relative Flux')
        ax1.get_xaxis().set_ticks([])
        ax1.set_ylim(ylim)
        ax1.set_xlim(wv1)
        ax1.plot(wv1, [0.]*2, '--', color='gray')

        # Confidence
        axc = plt.subplot(gs[1,ss])
        axc.plot(full_lam[gdi], np.minimum(sightline.prediction.offset_conv_sum[gdix],1), color='blue')
        axc.scatter(full_lam[gdi], sightline.prediction.offset_hist[gdix], s=6, color='green')
        axc.set_ylabel('Confidence')
        axc.set_xlabel('Wavelength')
        set_fontsize(axc, lsz)
        axc.set_xlim(wv1)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    outfile = 'fig_dla_confidence.pdf'
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))


def fig_dla_nhi(plate=4484, fiber=364):
    """ Plot the NHI values and fits for 2 DLAs
    """
    # Generate ID, load, and process the Sightline
    dr12_id = Id_DR12.from_csv(plate, fiber)
    sightline = read_sightline(dr12_id)
    sightline.process(default_model)
    # Generate model
    full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(sightline.loglam,
                                                              sightline.z_qso, REST_RANGE)
    lam_rest = full_lam_rest[full_ix_dla_range]
    peaks_offset = sightline.prediction.peaks_ixs

    # Start the plot
    fig = plt.figure(figsize=(6, 6))
    plt.clf()
    gs = gridspec.GridSpec(2,2)
    lsz = 13.

    # Loop on DLAs
    # DLA 1
    for ss in range(2):
        if ss == 0:
            wv1 = (3925., 4025.)
            ylim = (-0.9, 5.2)
        else:
            wv1 = (4100., 4240.)
        iwv = (full_lam > wv1[0]) & (full_lam < wv1[1])
        ixwv = (full_lam[full_ix_dla_range] > wv1[0]) & (full_lam[full_ix_dla_range] < wv1[1])
        gdi = np.where(full_ix_dla_range & iwv)[0]
        gdix = np.where(ixwv)[0]

        # Flux
        ax1 = plt.subplot(gs[0,ss])
        ax1.plot(full_lam[gdi], sightline.flux[gdi], '-k', lw=1.3, drawstyle='steps-mid')
        set_fontsize(ax1, lsz)
        ax1.set_ylabel('Relative Flux')
        ax1.get_xaxis().set_ticks([])
        ax1.set_ylim(ylim)
        ax1.set_xlim(wv1)
        ax1.plot(wv1, [0.]*2, '--', color='gray')

        # Voigt
        peak = peaks_offset[ss]
        dla_z = lam_rest[peak] * (1 + sightline.z_qso) / 1215.67 - 1
        density_pred_per_this_dla, mean_col_density_prediction, std_col_density_prediction, bias_correction = \
            sightline.prediction.get_coldensity_for_peak(peak)
        absorber = dict(z_dla=dla_z, column_density=mean_col_density_prediction)

        voigt_wave, voigt_model, ixs_mypeaks = generate_voigt_model(sightline, absorber)
        #ax1.plot(full_lam[ixs_mypeaks], sightline.flux[ixs_mypeaks], '+', mew=5, ms=10, color='green', alpha=1)
        ax1.plot(voigt_wave, voigt_model, 'b', lw=2.0)

        # NHI
        axN = plt.subplot(gs[1,ss])
        #axc.plot(full_lam[gdi], np.minimum(sightline.prediction.offset_conv_sum[gdix],1), color='blue')
        axN.scatter(full_lam[full_ix_dla_range][peak-30:peak+30],
                    density_pred_per_this_dla, s=6, color='blue')
        # Mean/sigma
        axN.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        axN.plot(wv1, [mean_col_density_prediction]*2, 'g--')
        axN.fill_between(wv1, [mean_col_density_prediction+std_col_density_prediction]*2,
                         [mean_col_density_prediction-std_col_density_prediction]*2,
                         color='green', alpha=0.3)
        axN.set_ylabel(r'$\log N_{\rm HI}$')
        axN.set_xlabel('Wavelength')
        set_fontsize(axN, lsz)
        axN.set_xlim(wv1)


    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    outfile = 'fig_dla_nhi.pdf'
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))



def fig_dla_injection(idla=11):
    outfile = 'fig_dla_injection.pdf'


    # DR5 sightlines (without DLAs)
    sdss = DLASurvey.load_SDSS_DR5(sample='all')
    slines, sdict = tset.grab_sightlines(sdss, flg_bal=0)
    igmsp = IgmSpec()

    # Open training
    test_file = CNN_result_path + 'gensample_hdf5_files/test_dlas_96629_10000.json'
    test_dlas = ltu.loadjson(test_file)
    sl = test_dlas[str(idla)]['sl']
    ispec, meta = igmsp.spectra_from_coord((slines['RA'][sl], slines['DEC'][sl]),
                                           groups=['SDSS_DR7'])

    test_spec = test_file.replace('json', 'hdf5')
    spec = lsio.readspec(test_spec, masking='edges')
    spec.select = idla

    # Name
    coord = ltu.radec_to_coord((slines['RA'][sl], slines['DEC'][sl]))
    jname = ltu.name_from_coord(coord)

    # Start the plot
    fig = plt.figure(figsize=(8, 5))
    plt.clf()
    gs = gridspec.GridSpec(2,1)
    xlim = (3820., 4750)
    ylim = (-2., 18.)

    # Real spectrum
    ax = plt.subplot(gs[0])
    ax.plot(ispec.wavelength, ispec.flux, 'k-', lw=1.2, drawstyle='steps-mid')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel('Relative Flux')
    ax.get_xaxis().set_ticks([])
    ylbl = 0.9
    tsz = 15.
    ax.text(0.50, ylbl, jname, transform=ax.transAxes, size=tsz, ha='center')
    set_fontsize(ax, 15.)

    # Mock spectrum
    ax2 = plt.subplot(gs[1])
    ax2.plot(spec.wavelength, spec.flux, 'k-', lw=1.2, drawstyle='steps-mid')

    # Axes
    #ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax2.set_xlabel('Wavelength (Ang)')
    ax2.set_ylabel('Relative Flux')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.text(0.5, ylbl, 'Mock '+jname, transform=ax2.transAxes,
             size=tsz, ha='center', color='blue')

    set_fontsize(ax2, 15.)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))


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
    pdb.set_trace()

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
        inax.plot(full_lam, sightline.flux, 'k', lw=1.2, drawstyle='steps-mid')
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


def fig_n09_no_detect(ml_dlasurvey=None):
    outfil='fig_n09_no_detect.pdf'
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


def fig_not_in_n09(ml_dlasurvey=None):
    """ DLAs in the ML but not in N07 (and mainly not in DR5 either)
    """
    outfil='fig_not_in_n09.pdf'
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


def fig_dr5_vs_ml(ml_dlasurvey=None):
    """ Plot Dz and DNHI for overlapping DLAs in DR5 vs. ML
    """
    outfil='fig_dr5_vs_ml.pdf'
    # Load DLA samples
    if ml_dlasurvey is None:
        _, ml_dlasurvey = load_ml_dr7()
    # Load DR5
    dr5 = DLASurvey.load_SDSS_DR5()  # This is the statistical sample
    # Load vette file
    vette_file = '../Vetting/vette_dr5.json'
    vdr5 = ltu.loadjson(vette_file)
    in_ml = np.array(vdr5['in_ml'])
    dr5_ml_idx = np.array(vdr5['dr5_idx'])
    # Cut down
    dr5.sightlines = dr5.sightlines[in_ml]
    new_mask = dla_stat(dr5, dr5.sightlines) # 737 good DLAs
    dr5.mask = new_mask
    dr5_dla_coord = dr5.coord
    dr5_dla_zabs = dr5.zabs
    dr5_dla_NHI = dr5.NHI
    assert len(dr5_dla_coord) == 737

    # Start the plot
    plt.figure(figsize=(4, 8))
    plt.clf()
    gs = gridspec.GridSpec(2,1)

    # dz
    ax1 = plt.subplot(gs[0])
    gdm = dr5_ml_idx >= 0
    iLLS = dr5_ml_idx == -9
    print('There are {:d} DLAs in DR5'.format(np.sum(dr5.NHI >= 20.299)))
    print('There are {:d} DLAs that matched to ML DLAs'.format(np.sum(gdm)))
    print('There are {:d} DLAs that matched to ML LLS'.format(np.sum(iLLS)))
    dz = dr5_dla_zabs[gdm]-ml_dlasurvey.zabs[dr5_ml_idx[gdm]]
    print('median dz = {}, std dz = {}'.format(np.median(dz), np.std(dz)))
    ax1.hist(dz, bins=50)
    # Axes
    #ax1.set_yscale("log", nonposy='clip')
    #ax1.set_ylim(1., 3000.)
    ax1.set_xlim(-0.03, 0.03)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax1.set_xlabel(r'$\Delta \, z$')
    ax1.set_ylabel('N')

    dNHI = dr5_dla_NHI[gdm]-ml_dlasurvey.NHI[dr5_ml_idx[gdm]]
    print('median dNHI = {}, std dNHI = {}'.format(np.median(dNHI), np.std(dNHI)))
    ax2 = plt.subplot(gs[1])
    ax2.hist(dNHI, bins=20)
    # Axes
    #ax1.set_yscale("log", nonposy='clip')
    #ax1.set_ylim(1., 3000.)
    #ax2.set_xlim(-0.02, 0.02)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.4))
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


def fig_n09_vs_ml(ml_dlasurvey=None):
    """ Plot Dz and DNHI for overlapping DLAs in PN vs. ML
    """
    outfil='fig_n09_vs_ml.pdf'
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


def fig_s2n_nhi_confidence(ml_dlasurvey=None):
    """ Plot Dz and DNHI for overlapping DLAs in DR5 vs. ML
    """
    outfil='fig_confidence.pdf'
    # Load DLA samples
    if ml_dlasurvey is None:
        _, ml_dlasurvey = load_ml_dr7()

    # Parse
    s2n = ml_dlasurvey.s2n
    confidence = ml_dlasurvey.confidence
    NHI = ml_dlasurvey.NHI

    # Start the plot
    fig = plt.figure(figsize=(6, 5))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])
    # Plot
    cm = plt.get_cmap('jet')
    cax = ax.scatter(s2n, NHI, s=2., c=confidence, cmap=cm)
    cb = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cb.set_label('Confidence')

    # Axes
    #ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel('S/N')
    ax.set_ylabel(r'$\log N_{\rm HI}$')
    ax.set_xlim(0.6, 200)
    ax.set_xscale("log", nonposy='clip')

    set_fontsize(ax, 15.)

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
        fig_n09_vs_ml()

    # Plot missed DLAs from N07
    if flg_fig & (2**1):
        fig_n09_no_detect()

    # Varying confidence DLAs
    if flg_fig & (2**2):
        fig_varying_confidence()

    # Varying confidence DLAs
    if flg_fig & (2**3):
        fig_ignore_flux()

    # Compare dz and dNHI between DR5 and ML
    if flg_fig & (2**4):
        fig_dr5_vs_ml()

    # Confidence
    if flg_fig & (2**5):
        fig_s2n_nhi_confidence()

    # DLA injection
    if flg_fig & (2**6):
        fig_dla_injection()

    # DLA injection
    if flg_fig & (2**7):
        fig_labels()

    # Confidence
    if flg_fig & (2**8):
        fig_dla_confidence()

    # NHI
    if flg_fig & (2**9):
        fig_dla_nhi()


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2**0   # dz, dNHI from N07 to ML
        #flg_fig += 2**1   # Missed DLAs in N07
        #flg_fig += 2**2   # Two DLAs with differing confidence
        #flg_fig += 2**3   # DLAs that ignored bad flux
        #flg_fig += 2**4   # DR5 dNHI and dz
        #flg_fig += 2**5   # Confidence vs. NHI and S/N
        flg_fig += 2**6   # DLA injection
        #flg_fig += 2**7   # CNN Labels
        #flg_fig += 2**8   # DLA confidence
        #flg_fig += 2**9   # DLA NHI
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
