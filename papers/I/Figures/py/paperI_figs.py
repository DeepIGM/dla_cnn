""" Module for the figures of FRB DLAs paper
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


from astropy import units as u
from astropy.table import Table

from linetools import utils as ltu


# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
sys.path.append(os.path.abspath("../Vetting/py"))
from vette_dr7 import load_ml_dr7


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


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2**0   # dz, dNHI from N07 to ML
        flg_fig += 2**1   # Missed DLAs in N07
        #flg_fig += 2**2   # g(z)
        #flg_fig += 2**3   # ne/nH
        #flg_fig += 2**4   # Average DM
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
