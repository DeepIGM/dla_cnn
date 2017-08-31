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
from astropy.coordinates import SkyCoord

from linetools import utils as ltu
from linetools.spectra import io as lsio
from linetools.analysis import voigt
from linetools.spectralline import AbsLine

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
    from dla_cnn.data_model.Id_GENSAMPLES import Id_GENSAMPLES
    from dla_cnn.absorption import generate_voigt_model, voigt_from_sightline
from dla_cnn.io import load_ml_dr7, load_ml_dr12, load_garnett16
from dla_cnn.catalogs import match_boss_catalogs
from dla_cnn import training_set as tset


# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
sys.path.append(os.path.abspath("../Vetting/py"))
from vette_test import pred_to_tbl, test_to_tbl

default_model = resource_filename('dla_cnn', "models/model_gensample_v7.1")

def init_for_ipython():
    #from imp import reload
    #import paperI_figs as pfigs
    _, ml_dlasurvey = load_ml_dr7()
    return ml_dlasurvey

def fig_ignore_flux(fsz=12.):  # Previous Fig 13
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
    NHI = sightline.prediction.density_data * loc_conf
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
    ylim = (19., 21.0)
    axN.set_ylim(ylim)

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
        axc.scatter(full_lam[gdi], sightline.prediction.offset_hist[gdix], s=6, color='green')
        axc.plot(full_lam[gdi], np.minimum(sightline.prediction.offset_conv_sum[gdix],1), color='blue')
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
    ax2.set_xlim(-1.3, 1.3)
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


def fig_test_nhi():
    """ compare injected vs. Predicted NHI values using 5k
     And overlay the fit
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    outfile = 'fig_test_nhi.pdf'
    # Load ML
    ml_abs = pred_to_tbl('../Vetting/data/test_dlas_5k96451_predictions.json.gz')
    # Load Test
    test_dlas = test_to_tbl('../Vetting/data/test_dlas_5k96451.json.gz')
    # Load vette
    vette_5k = ltu.loadjson('../Vetting/vette_5k.json')

    # Scatter plot of NHI
    test_ml_idx = np.array(vette_5k['test_idx'])
    any_abs = test_ml_idx != -99999
    #dz = ml_abs['zabs'][test_ml_idx[match]] - test_dlas['zabs'][match]
    abs_idx = np.abs(test_ml_idx)

    # Grab columns
    pred_NHI = ml_abs['NHI'][abs_idx[any_abs]] - ml_abs['biasNHI'][abs_idx[any_abs]]
    true_NHI = test_dlas['NHI'][any_abs]


    # Ridge regression & plot
    # p = np.polyfit(x,y,degree)
    degree, alpha = 3, 1
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
    model.fit(pred_NHI.reshape(-1,1), true_NHI)
    rval = np.linspace(20.0, 22.25, 1000)
    r_pred = model.predict(rval.reshape(-1, 1))
    # plt.plot(r, r_pred, linewidth=2, color='green')
    #plt.plot(r, np.polyval(p, r), linewidth=2, color='green')

    # Get polynomial from the model
    p = np.flipud(model.get_params()['ridge'].coef_)
    p[-1] += model.get_params()['ridge'].intercept_
    np.set_printoptions(precision=52)
    print(p)
    print(np.polyval(p, 21.0))

    # Start the plot
    fig = plt.figure(figsize=(5, 5))
    plt.clf()
    gs = gridspec.GridSpec(1,1)
    #xlim = (3820., 4750)
    #ylim = (-2., 18.)

    ax = plt.subplot(gs[0])
    ax.scatter(pred_NHI, true_NHI, s=0.1)

    # One-to-one line
    ax.plot(rval, rval, ':', color='gray')

    # Fit
    ax.plot(rval, r_pred, 'b--')

    ax.set_xlabel(r'Predicted $\log \, N_{\rm HI}$ (Uncorrected)')
    ax.set_ylabel(r'True $\log \, N_{\rm HI}$')
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    #ax.set_xlim(0.6, 200)


    set_fontsize(ax, 15.)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))


def fig_test_false_neg():
    """ Figure showing NHI and XXX for test 10k false negatives
    """
    outfile = 'fig_test_false_neg.pdf'

    # Load ML
    ml_abs = pred_to_tbl('../Vetting/data/test_dlas_96629_predictions.json.gz')
    # Load Test
    test_dlas = test_to_tbl('../Vetting/data/test_dlas_96629_10000.json.gz')
    # Load vette
    vette_10k = ltu.loadjson('../Vetting/vette_10k.json')
    test_ml_idx = np.array(vette_10k['test_idx'])

    # False neg

    # Start the plot
    fig = plt.figure(figsize=(6, 6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])

    # All True
    cm = plt.get_cmap('Greys')
    ax.hist2d(test_dlas['NHI'], test_dlas['zabs'], bins=20, cmap=cm)

    # False negatives - SLLS
    sllss = np.where((test_ml_idx < 0) & (test_ml_idx != -99999))[0]
    ax.scatter(test_dlas['NHI'][sllss], test_dlas['zabs'][sllss], color='blue', s=5.0, label='SLLS')

    # False negatives - Real Misses
    misses = np.where(test_ml_idx == -99999)[0]
    ax.scatter(test_dlas['NHI'][misses], test_dlas['zabs'][misses], marker='s', color='red', s=5.0, label='Missed')

    ax.set_xlabel(r'True $\log \, N_{\rm HI}$')
    ax.set_ylabel(r'$z_{\rm DLA}$')
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    #ax.set_xlim(0.6, 200)
    set_fontsize(ax, 15.)

    legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
                      handletextpad=0.3, fontsize='x-large', numpoints=1)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))


def fig_test_fneg_z():
    """ Examine redshifts of false neg
    """
    outfile = 'fig_test_fneg_z.pdf'

    # Load Test
    test_dlas = test_to_tbl('../Vetting/data/test_dlas_96629_10000.json.gz')
    # Load vette
    vette_10k = ltu.loadjson('../Vetting/vette_10k.json')
    test_ml_idx = np.array(vette_10k['test_idx'])

    # False neg

    # Start the plot
    fig = plt.figure(figsize=(6, 6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])

    # All True
    cm = plt.get_cmap('Greys')
    ax.hist(test_dlas['zabs'], bins=50)#, cmap=cm)

    # Misses
    misses = np.where(test_ml_idx == -99999)[0]
    ax.hist(test_dlas['zabs'][misses], color='black', bins=20)#, cmap=cm)
    '''
    # False negatives - SLLS
    sllss = np.where((test_ml_idx < 0) & (test_ml_idx != -99999))[0]
    ax.scatter(test_dlas['NHI'][sllss], test_dlas['zabs'][sllss], color='blue', s=5.0, label='SLLS')

    # False negatives - Real Misses
    misses = np.where(test_ml_idx == -99999)[0]
    ax.scatter(test_dlas['NHI'][misses], test_dlas['zabs'][misses], marker='s', color='red', s=5.0, label='Missed')
    '''

    ax.set_ylabel(r'N')
    ax.set_xlabel(r'$z_{\rm DLA}$')
    #ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    #ax.set_xlim(0.6, 200)
    set_fontsize(ax, 15.)

    #legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
    #                  handletextpad=0.3, fontsize='x-large', numpoints=1)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))



def fig_conf_vs_compl():
    """ Examine Completeness vs. Confidence for Test 10k
    """
    outfile = 'fig_conf_vs_compl.pdf'

    # Load Test
    test_dlas = test_to_tbl('../Vetting/data/test_dlas_96629_10000.json.gz')
    # Load vette
    vette_10k = ltu.loadjson('../Vetting/vette_10k.json')
    test_ml_idx = np.array(vette_10k['test_idx'])
    # Load ML
    ml_abs = pred_to_tbl('../Vetting/data/test_dlas_96629_predictions.json.gz')

    # Matches
    match = test_ml_idx >= 0
    conf = ml_abs['conf'][test_ml_idx[match]]
    max_compl = np.sum(match) / len(test_dlas)

    # Sort
    isrt = np.argsort(conf)

    # Start the plot
    fig = plt.figure(figsize=(6, 6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])

    # Plot
    cumsum = np.arange(np.sum(match)) / len(test_dlas)
    ax.plot(conf[isrt], max_compl - cumsum)

    ax.set_ylabel(r'Completeness (> conf)')
    ax.set_xlabel(r'Confidence')
    #ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.set_ylim(0.8, 1)
    set_fontsize(ax, 15.)

    #legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
    #                  handletextpad=0.3, fontsize='x-large', numpoints=1)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))

def fig_test_neg_overlap(ytxt=0.8):
    outfile = 'fig_test_neg_overlap.pdf'

    # Load Test
    test_dlas = test_to_tbl('../Vetting/data/test_dlas_96629_10000.json.gz')

    # Run the sightline!

    hdf5_datafile = CNN_result_path+'gensample_hdf5_files/test_dlas_96629_10000.hdf5'
    json_datafile = CNN_result_path+'gensample_hdf5_files/test_dlas_96629_10000.json'


    # Start the plot
    fig = plt.figure(figsize=(8, 5))
    plt.clf()
    gs = gridspec.GridSpec(2,1)

    # 2 -> 1
    idv, slv = 2929, 139
    G_id = Id_GENSAMPLES(idv, hdf5_datafile, json_datafile, sightlineid=slv)
    sightline = read_sightline(G_id)
    sightline.process(default_model)
    full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)

    # Real spectrum
    ax = plt.subplot(gs[0])
    xlim = (4050., 4500)
    ylim = (-2., 16.)
    ax.plot(full_lam, sightline.flux, 'k-', lw=1.2, drawstyle='steps-mid')
    ax.plot(xlim, [0.]*2, '--', color='gray')

    # Add 'truth' lines
    mt_id = test_dlas['ids'] == idv
    tsz = 12.
    for row in test_dlas[mt_id]:
        lya_wv = (row['zabs']+1)*1215.67
        ax.plot([lya_wv]*2, ylim, ':', color='green')
        ax.text(lya_wv, ylim[1]*ytxt, r'$\log N_{\rm HI} = $'+'{:0.2f}'.format(row['NHI']),
                color='green', size=tsz, rotation=90.)

    # Add voigt
    voigt_wave, voigt_model, ixs_mypeaks = voigt_from_sightline(sightline, 0)
    ax.plot(voigt_wave, voigt_model, 'b', lw=2.0)

    idla = 0
    lya_wv = sightline.dlas[idla]['spectrum']
    NHI = sightline.dlas[idla]['column_density']
    ax.plot([lya_wv]*2, ylim, '--', color='blue')
    ax.text(lya_wv, ylim[1]*ytxt, r'$\log N_{\rm HI} = $'+'{:0.2f}'.format(NHI),
            color='blue', size=tsz, rotation=90.)


    # Axes
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel('Relative Flux')
    ylbl = 0.9
    tsz = 15.
    set_fontsize(ax, 15.)
    #ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel('Wavelength (Ang)')

    # 1 -> 2
    idv, slv = 6161, 1782
    G_id = Id_GENSAMPLES(idv, hdf5_datafile, json_datafile, sightlineid=slv)
    sightline = read_sightline(G_id)
    sightline.process(default_model)
    full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)

    # Real spectrum
    ax = plt.subplot(gs[1])
    ax.plot(full_lam, sightline.flux, 'k-', lw=1.2, drawstyle='steps-mid')
    xlim = (5800., 6300)
    ylim = (-2., 9.)
    ax.plot(xlim, [0.]*2, '--', color='gray')

    # Add 'truth' lines
    mt_id = test_dlas['ids'] == idv
    tsz = 12.
    for row in test_dlas[mt_id]:
        tlya_wv = (row['zabs'] + 1) * 1215.67
        ax.plot([tlya_wv] * 2, ylim, ':', color='green')
        ax.text(tlya_wv, ylim[1] * ytxt, r'$\log N_{\rm HI} = $' + '{:0.2f}'.format(row['NHI']),
                color='green', size=tsz, rotation=90.)

    # Add voigt
    for idla in [1,2]:
        voigt_wave, voigt_model, ixs_mypeaks = voigt_from_sightline(sightline, idla)
        if idla == 1:
            pix = voigt_wave < tlya_wv
        else:
            pix = voigt_wave > tlya_wv
        ax.plot(voigt_wave[pix], voigt_model[pix], 'b', lw=2.0)

        lya_wv = sightline.dlas[idla]['spectrum']
        NHI = sightline.dlas[idla]['column_density']
        ax.plot([lya_wv] * 2, ylim, '--', color='blue')
        ax.text(lya_wv, ylim[1] * ytxt, r'$\log N_{\rm HI} = $' + '{:0.2f}'.format(NHI),
                color='blue', size=tsz, rotation=90.)

    # Axes
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel('Relative Flux')
    ylbl = 0.9
    tsz = 15.
    set_fontsize(ax, 15.)
    # ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel('Wavelength (Ang)')

    # ############
    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))


def fig_test_false_pos(ytxt=0.8):
    outfile = 'fig_test_false_pos.pdf'

    # Load Test
    test_dlas = test_to_tbl('../Vetting/data/test_dlas_96629_10000.json.gz')

    # Run the sightline!
    hdf5_datafile = CNN_result_path+'gensample_hdf5_files/test_dlas_96629_10000.hdf5'
    json_datafile = CNN_result_path+'gensample_hdf5_files/test_dlas_96629_10000.json'

    # Start the plot
    fig = plt.figure(figsize=(5, 8))
    plt.clf()
    gs = gridspec.GridSpec(3,1)

    for ss, ymin, ymax, idv, ipeak, idla in zip(range(3), [-2., -1., -2.],
                                                [7.5, 4., 6.],
                                          [97, 271, 2145], [1,1,1], [1,0,1]):
        G_id = Id_GENSAMPLES(idv, hdf5_datafile, json_datafile)
        sightline = read_sightline(G_id)
        sightline.process(default_model)
        full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)

        # Real spectrum
        ax = plt.subplot(gs[ss])
        ax.plot(full_lam, sightline.flux, 'k-', lw=1.2, drawstyle='steps-mid')

        # Add voigt
        voigt_wave, voigt_model, ixs_mypeaks = voigt_from_sightline(sightline, ipeak)
        ax.plot(voigt_wave, voigt_model, 'b', lw=2.0)

        idla = 0
        lya_wv = sightline.dlas[idla]['spectrum']
        xlim = (lya_wv-80., lya_wv+80.)
        NHI = sightline.dlas[idla]['column_density']
        ax.text(0.5, 0.9, r'$\log \, N_{\rm HI} = $'+'{:0.2f}'.format(NHI),
                color='blue', size=12., transform=ax.transAxes, ha='center')
        # Axes
        ax.set_xlim(xlim)
        ax.plot(xlim, [0.]*2, '--', color='gray', lw=0.5)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel('Relative Flux')
        set_fontsize(ax, 15.)
        ax.xaxis.set_major_locator(plt.MultipleLocator(50.))
        ax.set_xlabel('Wavelength (Ang)')

    # ############
    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))


def fig_test_low_s2n(ytxt=0.8):
    outfile = 'fig_test_low_s2n.pdf'

    # Load Test
    test_dlas = test_to_tbl('../Vetting/data/test_dlas_96629_10000.json.gz')

    # Run the sightline!
    hdf5_datafile = CNN_result_path+'gensample_hdf5_files/test_dlas_96629_10000.hdf5'
    json_datafile = CNN_result_path+'gensample_hdf5_files/test_dlas_96629_10000.json'

    # Start the plot
    fig = plt.figure(figsize=(5, 8))
    plt.clf()
    gs = gridspec.GridSpec(2,1)

    for ss, ymin, ymax, idv, ipeak, idla in zip(range(2), [-1., -5.],
                                                [2., 10.5],
                                          [334, 1279], [0,0], [0,0]):
        G_id = Id_GENSAMPLES(idv, hdf5_datafile, json_datafile)
        sightline = read_sightline(G_id)
        sightline.process(default_model)
        full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)

        # Real spectrum
        ax = plt.subplot(gs[ss])
        ax.plot(full_lam, sightline.flux, 'k-', lw=1.2, drawstyle='steps-mid')

        # Add voigt
        voigt_wave, voigt_model, ixs_mypeaks = voigt_from_sightline(sightline, ipeak)
        ax.plot(voigt_wave, voigt_model, 'b', lw=2.0)

        idla = 0
        lya_wv = sightline.dlas[idla]['spectrum']
        xlim = (lya_wv-90., lya_wv+90.)
        NHI = sightline.dlas[idla]['column_density']
        ax.text(0.5, 0.9, r'$\log \, N_{\rm HI} = $'+'{:0.2f}'.format(NHI),
                color='blue', size=14., transform=ax.transAxes, ha='center')
        # Axes
        ax.set_xlim(xlim)
        ax.plot(xlim, [0.]*2, '--', color='gray', lw=0.5)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel('Relative Flux')
        set_fontsize(ax, 15.)
        ax.xaxis.set_major_locator(plt.MultipleLocator(50.))
        ax.set_xlabel('Wavelength (Ang)')

    # ############
    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))

def fig_boss_hist(dztoler=0.015):
    """ Match ML to Garnett and compare dz and dNHI
    """
    outfil='fig_boss_hist.pdf'

    # Load BOSS ML
    _, dr12_abs = load_ml_dr12()
    # Cut on DLA
    dlas = dr12_abs['NHI'] >= 20.3
    no_bals = dr12_abs['flg_BAL'] == 0
    high_conf = dr12_abs['conf'] > 0.9
    zem = dr12_abs['zem'] > dr12_abs['zabs']
    zcut = dr12_abs['zabs'] > 2.
    dr12_cut = dlas & no_bals & high_conf & zem & zcut
    dr12_dla = dr12_abs[dr12_cut]

    # Load Garnett
    g16_abs = load_garnett16()
    g16_dlas = g16_abs[g16_abs['log.NHI'] >= 20.3]

    # Match
    dr12_to_g16 = match_boss_catalogs(dr12_dla, g16_dlas, dztoler=dztoler)
    matched = dr12_to_g16 >= 0
    g16_idx = dr12_to_g16[matched]
    print("We matched {:d} of {:d} DLAs between high quality ML and G16 within dz={:g}".format(
        np.sum(matched), np.sum(dr12_cut), dztoler))

    high_conf = (dr12_dla['conf'][matched] > 0.9) & (g16_dlas['pDLAD'][g16_idx] > 0.9)
    print("Of these, {:d} are high confidence in both".format(np.sum(high_conf)))

    # Start the plot
    plt.figure(figsize=(5, 8))
    plt.clf()
    gs = gridspec.GridSpec(2,1)

    # dz
    dz = dr12_dla['zabs'][matched] - g16_dlas['z_DLA'][g16_idx]
    ax1 = plt.subplot(gs[0])
    print('median dz = {}, std dz = {}'.format(np.median(dz), np.std(dz)))
    ax1.hist(dz, bins=50)
    # Axes
    #ax1.set_yscale("log", nonposy='clip')
    #ax1.set_ylim(1., 3000.)
    ax1.set_xlim(-0.03, 0.03)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax1.set_xlabel(r'$\Delta \, z$')
    ax1.set_ylabel('N')

    dNHI = dr12_dla['NHI'][matched] - g16_dlas['log.NHI'][g16_idx]
    print('median dNHI = {}, std dNHI = {}'.format(np.median(dNHI), np.std(dNHI)))
    ax2 = plt.subplot(gs[1])
    ax2.hist(dNHI, bins=20)
    # Axes
    #ax1.set_yscale("log", nonposy='clip')
    #ax1.set_ylim(1., 3000.)
    ax2.set_xlim(-1.2, 1.2)
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


def fig_boss_dNHI(dztoler=0.015):
    """ Match ML to Garnett and compare dNHI in a scatter plot
    """
    outfil='fig_boss_dNHI.pdf'

    # Load BOSS ML
    _, dr12_abs = load_ml_dr12()
    # Cut on DLA
    dlas = dr12_abs['NHI'] >= 20.3
    dr12_dla = dr12_abs[dlas]

    # Load Garnett
    g16_abs = load_garnett16()
    g16_dlas = g16_abs[g16_abs['log.NHI'] >= 20.3]

    # Match
    dr12_to_g16 = match_boss_catalogs(dr12_dla, g16_dlas, dztoler=dztoler)
    matched = dr12_to_g16 >= 0
    g16_idx = dr12_to_g16[matched]
    print("We matched {:d} DLAs between ML and G16 within dz={:g}".format(
        np.sum(matched), dztoler))

    high_conf = (dr12_dla['conf'][matched] > 0.9) & (g16_dlas['pDLAD'][g16_idx] > 0.9)
    print("Of these, {:d} are high confidence in both".format(np.sum(high_conf)))

    # Start the plot
    plt.figure(figsize=(6, 6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Scatter me

    NHI = dr12_dla['NHI'][matched]
    dNHI = dr12_dla['NHI'][matched] - g16_dlas['log.NHI'][g16_idx]
    print('median dNHI = {}, std dNHI = {}'.format(np.median(dNHI), np.std(dNHI)))
    ax = plt.subplot(gs[0])
    cm = plt.get_cmap('jet')
    cax = ax.scatter(NHI, dNHI, s=2., c=dr12_dla['zabs'][matched], cmap=cm)
    # Axes
    #ax1.set_yscale("log", nonposy='clip')
    #ax1.set_ylim(1., 3000.)
    #ax2.set_xlim(-0.02, 0.02)
    #ax.xaxis.set_major_locator(plt.MultipleLocator(0.4))
    ax.set_ylabel(r'$\Delta \, \log \, N_{\rm HI}$')
    ax.set_xlabel(r'$\log \, N_{\rm HI}$')

    # Legend
    #legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
    #                  handletextpad=0.3, fontsize='medium', numpoints=1)
    set_fontsize(ax,15.)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfil)
    plt.close()
    print("Wrote {:s}".format(outfil))


def fig_boss_badz():
    """ Scatter plot of highNHI from G16 with large but not too large dz
    """
    outfil='fig_boss_badz.pdf'

    # Read Vet file
    missing_g16 = Table.read('../Vetting/G16_highNHI_misses.ascii', format='ascii.fixed_width')

    # Cut on dz
    cutz = np.abs(missing_g16['dz12']) < 0.05
    print("There are {:d} high NHI DLAs from G16 with a large dz offset".format(np.sum(cutz)))

    # Start the plot
    fig = plt.figure(figsize=(6, 6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Scatter me

    NHI = missing_g16['log.NHI'][cutz]
    dNHI = missing_g16['dNHI'][cutz]
    zabs = missing_g16['z_DLA'][cutz]
    ax = plt.subplot(gs[0])
    cm = plt.get_cmap('jet')
    cax = ax.scatter(zabs, NHI, s=10., c=dNHI, cmap=cm)
    cb = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cb.set_label('dNHI (G16-ML)')
    # Axes
    #ax1.set_yscale("log", nonposy='clip')
    #ax1.set_ylim(1., 3000.)
    #ax2.set_xlim(-0.02, 0.02)
    #ax.xaxis.set_major_locator(plt.MultipleLocator(0.4))
    ax.set_ylabel(r'$\Delta \, \log \, N_{\rm HI}$')
    ax.set_xlabel(r'$\log \, N_{\rm HI}$')

    # Legend
    #legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
    #                  handletextpad=0.3, fontsize='medium', numpoints=1)
    set_fontsize(ax,15.)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfil)
    plt.close()
    print("Wrote {:s}".format(outfil))


def fig_boss_missing():
    """ Scatter plot of highNHI from G16 with no ML counterpart
    """
    outfil='fig_boss_missing.pdf'

    # Read Vet file
    missing_g16 = Table.read('../Vetting/G16_highNHI_misses.ascii', format='ascii.fixed_width')

    # Cut on dz
    missing = np.abs(missing_g16['dz12']) > 0.05
    print("There are {:d} high NHI DLAs from G16 not in ML".format(np.sum(missing)))

    # Load Garnett for stats
    NHImin = 21.8
    g16_abs = load_garnett16()
    g16_dlas = g16_abs[g16_abs['log.NHI'] >= 20.3]
    high_high = (g16_dlas['pDLAD'] > 0.9) & (g16_dlas['log.NHI'] > NHImin) & (
        g16_dlas['flg_BAL'] == 0) & (g16_dlas['z_DLA'] > 2.)
    print("There are {:d} NHI>{:g}, high confidence DLAs in G16 with z>2".format(np.sum(high_high), NHImin))
    high_missed = missing_g16['log.NHI'] > NHImin  # |dz| < 0.015
    print("Of these, {:d} do not match in ML within dz=0.015".format(np.sum(high_missed)))

    # Start the plot
    fig = plt.figure(figsize=(6, 6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Scatter me

    NHI = missing_g16['log.NHI'][missing]
    zabs = missing_g16['z_DLA'][missing]
    SNR = missing_g16['SNR'][missing]

    ax = plt.subplot(gs[0])
    cm = plt.get_cmap('jet')
    cax = ax.scatter(zabs, NHI, s=10., c=SNR, cmap=cm)
    cb = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cb.set_label('S/N')
    # Axes
    #ax1.set_yscale("log", nonposy='clip')
    #ax1.set_ylim(1., 3000.)
    #ax2.set_xlim(-0.02, 0.02)
    #ax.xaxis.set_major_locator(plt.MultipleLocator(0.4))
    ax.set_ylabel(r'$\Delta \, \log \, N_{\rm HI}$')
    ax.set_xlabel(r'$\log \, N_{\rm HI}$')

    # Legend
    #legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
    #                  handletextpad=0.3, fontsize='medium', numpoints=1)
    set_fontsize(ax,15.)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfil)
    plt.close()
    print("Wrote {:s}".format(outfil))


def fig_g16_junk():
    outfile='fig_g16_junk.pdf'

    junk_plates = [6466, 5059, 4072, 3969]
    junk_fibers = [740, 906, 162, 788]
    wvoffs = [200., 200., 200., 200.]
    zabs = [2.8159, 2.4272, 3.3509, 2.9467]
    NHI = [21.33, 22.08, 22.35, 21.78]
    show_DLA = [False, True, True, True]
    DLA_conti = [0., 3., 1.5, 0.6]
    ylim = [(0.,5.), (-1., 4.), (-0.5,1.7), (-0.7,1.3)]

    igmsp = IgmSpec()
    meta = igmsp['BOSS_DR12'].meta

    # Start the plot
    fig = plt.figure(figsize=(5, 8))
    plt.clf()
    gs = gridspec.GridSpec(4,1)

    for ss in range(len(junk_plates)):
        ax = plt.subplot(gs[ss])
        plate, fiber, wvoff = junk_plates[ss], junk_fibers[ss], wvoffs[ss]
        imt = np.where((meta['PLATE'] == plate) & (meta['FIBERID'] == fiber))[0][0]
        # Load spec
        scoord = SkyCoord(ra=meta['RA_GROUP'][imt], dec=meta['DEC_GROUP'][imt], unit='deg')
        spec, _ = igmsp.spectra_from_coord(scoord, groups=['BOSS_DR12'])

        wv_lya = (1+zabs[ss])*1215.67
        xlim = (wv_lya-wvoff, wv_lya+wvoff)
        # Plot
        ax.plot(spec.wavelength, spec.flux, 'k-', lw=1.2, drawstyle='steps-mid')
        ax.plot(spec.wavelength, spec.sig, 'r:')
        ax.axvline(wv_lya, color='g', linestyle=':', lw=1.5)
        ax.plot(xlim, [0.]*2, '--', color='gray', lw=1.)

        # DLA?
        if show_DLA[ss]:
            lya = AbsLine(1215.67*u.AA, z=zabs[ss])
            lya.attrib['N'] = 10**NHI[ss] / u.cm**2
            lya.attrib['b'] = 20*u.km/u.s
            vmodel = voigt.voigt_from_abslines(spec.wavelength, lya)
            # Plot
            ax.plot(vmodel.wavelength, vmodel.flux*DLA_conti[ss], 'b--')

        if ss == 2:  #
            ax.axvline((1+3.28)*1215.67, color='purple', linestyle='-.', lw=1.5)
            ax.axvline((1+3.40)*1215.67, color='purple', linestyle='-.', lw=1.5)

        # Axes
        ax.set_ylim(ylim[ss])
        ax.set_xlim(xlim)
        #ax.xaxis.set_major_locator(plt.MultipleLocator(0.4))
        ax.set_ylabel(r'Flux')
        ax.set_xlabel(r'Wavelength (Ang)')
        #ax.text(0.5, 0.9, r'$\log \, N_{\rm HI} = $'+'{:0.2f}'.format(NHI),
        #        color='blue', size=14., transform=ax.transAxes, ha='center')

        set_fontsize(ax, 13.)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))


def fig_g16_good():
    """ G16 predicts a 10^22 that ML has as being considerably lower
    # Actually, the first one is better for ML
    """
    outfile='fig_g16_good.pdf'

    junk_plates = [7137,6139, 4374]
    junk_fibers = [194,598, 401]
    wvoffs = [250.,200., 250.]
    zabs = [2.2095, 2.0807, 2.3393]
    G16_NHI = [22.13, 21.80, 22.27]
    ML_NHI = [21.69, 21.34, 21.71]
    show_DLA = [True, True, True]
    DLA_conti = [7.,5., 5.]
    ylim = [(-1.,11.), (-1,11.), (-1,11.)]

    igmsp = IgmSpec()
    meta = igmsp['BOSS_DR12'].meta

    # Start the plot
    fig = plt.figure(figsize=(5, 8))
    plt.clf()
    gs = gridspec.GridSpec(4,1)

    for ss in range(len(junk_plates)):
        ax = plt.subplot(gs[ss])
        plate, fiber, wvoff = junk_plates[ss], junk_fibers[ss], wvoffs[ss]
        imt = np.where((meta['PLATE'] == plate) & (meta['FIBERID'] == fiber))[0][0]
        # Load spec
        scoord = SkyCoord(ra=meta['RA_GROUP'][imt], dec=meta['DEC_GROUP'][imt], unit='deg')
        spec, _ = igmsp.spectra_from_coord(scoord, groups=['BOSS_DR12'])

        wv_lya = (1+zabs[ss])*1215.67
        xlim = (wv_lya-wvoff, wv_lya+wvoff)
        # Plot
        ax.plot(spec.wavelength, spec.flux, 'k-', lw=1.2, drawstyle='steps-mid')
        ax.plot(spec.wavelength, spec.sig, 'r:')
        ax.axvline(wv_lya, color='g', linestyle=':', lw=1.5)
        ax.plot(xlim, [0.]*2, '--', color='gray', lw=1.)

        # DLA?
        if show_DLA[ss]:
            for ii in range(2):
                if ii == 0:
                    NHI = G16_NHI[ss]
                    clr = 'b'
                else:
                    NHI = ML_NHI[ss]
                    clr = 'g'
                lya = AbsLine(1215.67*u.AA, z=zabs[ss])
                lya.attrib['N'] = 10**NHI / u.cm**2
                lya.attrib['b'] = 20*u.km/u.s
                vmodel = voigt.voigt_from_abslines(spec.wavelength, lya)
                ax.plot(vmodel.wavelength, vmodel.flux*DLA_conti[ss], '--', color=clr)

        if ss == 2:  #
            ax.axvline((1+3.28)*1215.67, color='purple', linestyle='-.', lw=1.5)
            ax.axvline((1+3.40)*1215.67, color='purple', linestyle='-.', lw=1.5)

        # Axes
        ax.set_ylim(ylim[ss])
        ax.set_xlim(xlim)
        #ax.xaxis.set_major_locator(plt.MultipleLocator(0.4))
        ax.set_ylabel(r'Flux')
        ax.set_xlabel(r'Wavelength (Ang)')
        #ax.text(0.5, 0.9, r'$\log \, N_{\rm HI} = $'+'{:0.2f}'.format(NHI),
        #        color='blue', size=14., transform=ax.transAxes, ha='center')

        set_fontsize(ax, 13.)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))

def fig_dla_example():
    """ Example spectrum + DLA for Fig 1
    """
    outfile='fig_dla_example.pdf'

    plate = 1648
    fiber = 469

    igmsp = IgmSpec()
    meta = igmsp['SDSS_DR7'].meta

    # Start the plot
    fig = plt.figure(figsize=(8, 5))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])
    imt = np.where((meta['PLATE'] == plate) & (meta['FIBER'] == fiber))[0][0]
    # Load spec
    scoord = SkyCoord(ra=meta['RA_GROUP'][imt], dec=meta['DEC_GROUP'][imt], unit='deg')
    spec, _ = igmsp.spectra_from_coord(scoord, groups=['SDSS_DR7'])

    xlim = (3800., 4520.)
        # Plot
    ax.plot(spec.wavelength, spec.flux, 'k-', lw=1.2, drawstyle='steps-mid')
    ax.plot(spec.wavelength, spec.sig, 'r:')
    ax.plot(xlim, [0.]*2, '--', color='gray', lw=1.)


    # Axes
    ax.set_ylim(-3., 42)
    ax.set_xlim(xlim)
    #ax.xaxis.set_major_locator(plt.MultipleLocator(0.4))
    ax.set_ylabel(r'Relative Flux')
    ax.set_xlabel(r'Wavelength ($\AA$)')

    set_fontsize(ax, 15.)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))


def fig_new_dr7():
    outfile='fig_new_dr7.pdf'

    if False:
        _, dr7_dlas = load_ml_dr7()
        known = np.array([False]*dr7_dlas.nsys)
        vette_dr5 = ltu.loadjson('../Vetting/vette_dr5.json')
        dr5_ml_idx = np.array(vette_dr5['dr5_idx'])
        mdla = dr5_ml_idx >= 0
        known[dr5_ml_idx[mdla]] = True

        # N09
        vette_dr7_n09 = ltu.loadjson('../Vetting/vette_dr7_pn.json')
        n09_ml_idx = np.array(vette_dr7_n09['pn_idx'])
        mdla = n09_ml_idx >= 0
        known[n09_ml_idx[mdla]] = True

        # Generate a Table
        new = ~known
        new_tbl = Table()
        for key in ['plate', 'fiber', 'zem', 'zabs', 'NHI', 'confidence']:
            new_tbl[key] = getattr(dr7_dlas, key)[new]
        new_tbl.write("new_DR7_DLAs.ascii", format='ascii.fixed_width', overwrite=True)


    junk_plates = [278, 442, 1152]
    junk_fibers = [208, 508, 498]
    wvoffs = [150., 150., 200.]
    zabs = [2.93989, 2.8044, 2.6357]
    NHI = [20.727, 20.644, 21.25]
    show_DLA = [True, True, True]
    DLA_conti = [7.5, 3.7, 2.2]
    ylim = [(-1.,10.), (-1,7.), (-1,4.)]

    igmsp = IgmSpec()
    meta = igmsp['SDSS_DR7'].meta

    # Start the plot
    fig = plt.figure(figsize=(5, 8))
    plt.clf()
    gs = gridspec.GridSpec(3,1)

    for ss in range(len(junk_plates)):
        ax = plt.subplot(gs[ss])
        plate, fiber, wvoff = junk_plates[ss], junk_fibers[ss], wvoffs[ss]
        imt = np.where((meta['PLATE'] == plate) & (meta['FIBER'] == fiber))[0][0]
        # Load spec
        scoord = SkyCoord(ra=meta['RA_GROUP'][imt], dec=meta['DEC_GROUP'][imt], unit='deg')
        spec, _ = igmsp.spectra_from_coord(scoord, groups=['SDSS_DR7'])
        jname = ltu.name_from_coord(scoord)

        wv_lya = (1+zabs[ss])*1215.67
        xlim = (wv_lya-wvoff, wv_lya+wvoff)
        # Plot
        ax.plot(spec.wavelength, spec.flux, 'k-', lw=1.2, drawstyle='steps-mid')
        ax.plot(spec.wavelength, spec.sig, 'r:')
        ax.axvline(wv_lya, color='g', linestyle=':', lw=1.5)
        ax.plot(xlim, [0.]*2, '--', color='gray', lw=1.)

        # DLA?
        if show_DLA[ss]:
            lya = AbsLine(1215.67*u.AA, z=zabs[ss])
            lya.attrib['N'] = 10**NHI[ss] / u.cm**2
            lya.attrib['b'] = 20*u.km/u.s
            vmodel = voigt.voigt_from_abslines(spec.wavelength, lya)
            ax.plot(vmodel.wavelength, vmodel.flux*DLA_conti[ss], '--', color='b')

        # Axes
        ax.set_ylim(ylim[ss])
        ax.set_xlim(xlim)
        #ax.xaxis.set_major_locator(plt.MultipleLocator(0.4))
        ax.set_ylabel(r'Relative Flux')
        ax.set_xlabel(r'Wavelength (Ang)')
        ax.text(0.5, 0.9, '{:s}: '.format(jname)+r'$z_{\rm abs}=$'+'{:0.3f}'.format(
            zabs[ss])+r', $\log \, N_{\rm HI}=$'+'{:0.2f}'.format(NHI[ss]),
                color='blue', size=13., transform=ax.transAxes, ha='center')

        set_fontsize(ax, 13.)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))



def fig_g16_s2n_vs_NHI():
    """ Scatter plot of NHI vs. S/N from G16
    Highly confident systems only
    """
    outfil='fig_g16_s2n_vs_NHI.pdf'

    # Load Garnett for stats
    g16_abs = load_garnett16()
    g16_dlas = g16_abs[g16_abs['log.NHI'] >= 20.3]
    high_conf = (g16_dlas['pDLAD'] > 0.9) & (g16_dlas['flg_BAL'] == 0) & (g16_dlas['z_DLA'] > 2.) & (
        g16_dlas['SNR'] > 0.01)

    # Start the plot
    fig = plt.figure(figsize=(6, 6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Scatter me

    NHI = g16_dlas['log.NHI'][high_conf]
    SNR = g16_dlas['SNR'][high_conf]
    zabs = g16_dlas['z_DLA'][high_conf]

    SNRmin = np.min(SNR)
    SNRmax = np.max(SNR)
    SNRbins = 10**np.linspace(np.log10(SNRmin), np.log10(SNRmax), 20)
    avgN = np.zeros(SNRbins.size-1)
    iSNR = np.digitize(SNR, SNRbins) - 1
    for ii in range(avgN.size):
        idx = iSNR == ii
        #avgN[ii] = np.mean(NHI[idx])
        avgN[ii] = np.log10(np.mean(10**NHI[idx]))

    ax = plt.subplot(gs[0])
    cm = plt.get_cmap('jet')
    cax = ax.scatter(SNR, NHI, s=2., c=zabs, cmap=cm)
    cb = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cb.set_label('z')

    # AvgN
    ax.plot(SNRbins[:-1], avgN, 'k-')
    # Axes
    ax.set_xscale("log", nonposy='clip')
    #ax1.set_ylim(1., 3000.)
    ax.set_xlim(0.5, 100.)
    #ax.xaxis.set_major_locator(plt.MultipleLocator(0.4))
    ax.set_xlabel('S/N')
    ax.set_ylabel(r'$\log \, N_{\rm HI}$')

    # Legend
    #legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
    #                  handletextpad=0.3, fontsize='medium', numpoints=1)
    set_fontsize(ax,15.)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfil)
    plt.close()
    print("Wrote {:s}".format(outfil))



def fig_g16_improbable():
    outfile='fig_g16_improbable.pdf'

    junk_plates = [6466, 4799]
    junk_fibers = [740, 729]
    wvoffs = [200., 200.]
    zabs = [2.8159, 2.0767]
    NHI = [21.33, 22.21]

    igmsp = IgmSpec()
    meta = igmsp['BOSS_DR12'].meta

    # Start the plot
    fig = plt.figure(figsize=(5, 8))
    plt.clf()
    gs = gridspec.GridSpec(4,1)

    for ss in range(len(junk_plates)):
        ax = plt.subplot(gs[ss])
        plate, fiber, wvoff = junk_plates[ss], junk_fibers[ss], wvoffs[ss]
        imt = np.where((meta['PLATE'] == plate) & (meta['FIBERID'] == fiber))[0][0]
        # Load spec
        scoord = SkyCoord(ra=meta['RA_GROUP'][imt], dec=meta['DEC_GROUP'][imt], unit='deg')
        spec, _ = igmsp.spectra_from_coord(scoord, groups=['BOSS_DR12'])

        wv_lya = (1+zabs[ss])*1215.67
        xlim = (wv_lya-wvoff, wv_lya+wvoff)
        # Plot
        ax.plot(spec.wavelength, spec.flux, 'k-', lw=1.2, drawstyle='steps-mid')
        ax.axvline(wv_lya, color='g', linestyle='--', lw=1.5)

        # Axes
        #ax.set_ylim(1., 3000.)
        ax.set_xlim(xlim)
        #ax.xaxis.set_major_locator(plt.MultipleLocator(0.4))
        ax.set_ylabel(r'Flux')
        ax.set_xlabel(r'Wavelength (Ang)')
        #ax.text(0.5, 0.9, r'$\log \, N_{\rm HI} = $'+'{:0.2f}'.format(NHI),
        #        color='blue', size=14., transform=ax.transAxes, ha='center')

        set_fontsize(ax, 13.)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))



def fig_ml_boss_dlas():
    """ Figure showing NHI and zabs for ML-BOSS
    """
    outfile = 'fig_ml_boss_dlas.pdf'

    # Load ML
    _, boss_abs = load_ml_dr12()

    # Cut
    dlas = boss_abs['NHI'] >= 20.3
    no_bals = boss_abs['flg_BAL'] == 0
    high_conf = boss_abs['conf'] > 0.9
    zem = boss_abs['zem'] > boss_abs['zabs']
    zcut = boss_abs['zabs'] > 2.

    cut = dlas & no_bals & high_conf & zem & zcut
    boss_dlas = boss_abs[cut]
    print("There are {:d} DLAs in the high-quality ML BOSS sample".format(np.sum(cut)))

    # Start the plot
    fig = plt.figure(figsize=(7.5, 6))
    plt.clf()
    cm = plt.get_cmap('Blues')
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])

    nbin = 20
    xbins = np.linspace(2., 5., nbin)
    ybins = np.linspace(20.3, 22., nbin)
    counts, xedges, yedges = np.histogram2d(boss_dlas['zabs'],
        boss_dlas['NHI'], bins=(xbins, ybins))
    # Stretch counts
    counts = np.log10(np.maximum(counts,1.))
    #max_c = np.max(counts)
    pax = ax.pcolormesh(xedges, yedges, counts.transpose(), cmap=cm)#, vmin=0, vmax=max_c/5.)
    cb = fig.colorbar(pax, fraction=0.046, pad=0.04)
    cb.set_label(r'$\log_{10} \, n_{\rm DLA}$')
    # All True
    #ax.hist2d(boss_dlas['zabs'], boss_dlas['NHI'], bins=20, cmap=cm)

    # False negatives - SLLS
    #ax.scatter(test_dlas['NHI'][sllss], test_dlas['zabs'][sllss], color='blue', s=5.0, label='SLLS')


    ax.set_ylabel(r'$\log \, N_{\rm HI}$')
    ax.set_xlabel(r'$z_{\rm DLA}$')
    #ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    #ax.set_xlim(0.6, 200)
    set_fontsize(ax, 15.)

    legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
                      handletextpad=0.3, fontsize='x-large', numpoints=1)

    # Finish
    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    plt.savefig(outfile)
    plt.close()
    print("Wrote {:s}".format(outfile))

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

    # test 10k NHI
    if flg_fig & (2**10):
        fig_test_nhi()

    # test false neg
    if flg_fig & (2**11):
        fig_test_fneg_z()
        #fig_test_false_neg()

    # Overlap in test
    if flg_fig & (2**12):
        fig_test_neg_overlap()

    # False positives
    if flg_fig & (2**13):
        fig_test_false_pos()

    # Good IDs of low S/N
    if flg_fig & (2**14):
        fig_test_low_s2n()

    # BOSS histogram of matches
    if flg_fig & (2**15):
        fig_boss_hist()

    # BOSS matches dNHI scatter plot
    if flg_fig & (2**16):
        fig_boss_dNHI()

    # High NHI G16 with 0.015 < dz < 0.05
    if flg_fig & (2**17):
        fig_boss_badz()

    # High NHI G16 missing
    if flg_fig & (2**18):
        fig_boss_missing()

    # G16 junk
    if flg_fig & (2**19):
        fig_g16_junk()

    # G16 good
    if flg_fig & (2**20):
        fig_g16_good()

    # DLA example
    if flg_fig & (2**21):
        fig_dla_example()

    # New DR7 DLAs
    if flg_fig & (2**22):
        fig_new_dr7()

    # G16 S/N vs. NHI
    if flg_fig & (2**23):
        fig_g16_s2n_vs_NHI()

    # ML BOSS DLAs
    if flg_fig & (2**24):
        fig_ml_boss_dlas()

    # Conf vs. Completeness (Test)
    if flg_fig & (2**25):
        fig_conf_vs_compl()


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
        #flg_fig += 2**6   # DLA injection
        flg_fig += 2**7   # CNN Labels
        #flg_fig += 2**8   # DLA confidence
        #flg_fig += 2**9   # DLA NHI
        #flg_fig += 2**10   # Compare NHI in test 5k
        #flg_fig += 2**11   # False negatives in test 10k
        #flg_fig += 2**12   # Negative overlap
        #flg_fig += 2**13   # False positives
        #flg_fig += 2**14   # Test -- Good IDs of low S/N
        #flg_fig += 2**15   # Compare BOSS matches
        #flg_fig += 2**16   # BOSS dNHI scatter plot for matches
        #flg_fig += 2**17   # High NHI G16 with 0.015 < dz < 0.05
        #flg_fig += 2**18   # High NHI G16 that are simply missing
        #flg_fig += 2**19   # G16 junk
        #flg_fig += 2**20   # G16 good
        #flg_fig += 2**21   # DLA example (Fig 1)
        #flg_fig += 2**22   # New DLAs in DR7
        #flg_fig += 2**23   # G16 S/N vs. NHI
        #flg_fig += 2**24   # BOSS 2D Hist of DLAs
        flg_fig += 2**25   # Confidence vs. completeness
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
