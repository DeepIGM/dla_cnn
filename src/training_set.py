""" Module to vette results against Human catalogs
  SDSS-DR5 (JXP) and BOSS (Notredaeme)
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import pdb

from scipy import interpolate


from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
from astropy.table import Table
from astropy.io.fits import Header

from specdb.specdb import IgmSpec

from linetools import utils as ltu
from linetools.spectra.xspectrum1d import XSpectrum1D
from pyigm.surveys.dlasurvey import DLASurvey


def grab_sightlines(dlasurvey=None, flg_bal=None, zmin=2.3, s2n=5., DX=0.,
                    igmsp_survey='SDSS_DR7', update_zem=True):
    """ Grab a set of sightlines without DLAs from a DLA survey
    Insist that all have spectra occur in igmspec
    Update sightline zem with igmspec zem

    Parameters
    ----------
    dlas : DLASurvey
      Usually SDSS or BOSS
    flg_bal : int, optional
      Maximum BAL flag (0=No signature, 1=Weak BAL, 2=BAL)
    s2n : float, optional
      Minimum S/N as defined in some manner
    DX : float, optional
      Restrict on DX
    zmin : float, optional
      Minimum redshift for zem
    update_zem : bool, optional
      Update zem in sightlines?

    Returns
    -------
    final : Table
      astropy Table of good sightlines
    sdict : dict
      dict describing the sightlines
    """
    from specdb.cat_utils import match_ids
    igmsp = IgmSpec()
    # Init
    if dlasurvey is None:
        print("Using the DR5 sample for the sightlines")
        dlasurvey = DLASurvey.load_SDSS_DR5(sample='all')
        igmsp_survey = 'SDSS_DR7'
    nsight = len(dlasurvey.sightlines)
    keep = np.array([True]*nsight)
    meta = igmsp[igmsp_survey].meta

    # Avoid DLAs
    dla_coord = dlasurvey.coord
    sl_coord = SkyCoord(ra=dlasurvey.sightlines['RA'], dec=dlasurvey.sightlines['DEC'])
    idx, d2d, d3d = match_coordinates_sky(sl_coord, dla_coord, nthneighbor=1)
    clear = d2d > 1*u.arcsec
    keep = keep & clear

    # BAL
    if flg_bal is not None:
        gd_bal = dlasurvey.sightlines['FLG_BAL'] <= flg_bal
        keep = keep & gd_bal

    # S/N
    if s2n > 0.:
        gd_s2n = dlasurvey.sightlines['S2N'] > s2n
        keep = keep & gd_s2n

    # Cut on DX
    if DX > 0.:
        gd_DX = dlasurvey.sightlines['DX'] > DX
        keep = keep & gd_DX

    # igmsp
    qso_coord = SkyCoord(ra=meta['RA'], dec=meta['DEC'], unit='deg')
    idxq, d2dq, d3dq = match_coordinates_sky(sl_coord, qso_coord, nthneighbor=1)
    in_igmsp = d2dq < 1*u.arcsec
    keep = keep & in_igmsp

    # Check zem and dz
    #igm_id = meta['IGM_ID'][idxq]
    #cat_rows = match_ids(igm_id, igmsp.cat['IGM_ID'])
    #zem = igmsp.cat['zem'][cat_rows]
    zem = meta['zem'][idxq]
    dz = np.abs(zem - dlasurvey.sightlines['ZEM'])
    gd_dz = dz < 0.1
    keep = keep & gd_dz #& gd_zlim
    if zmin is not None:
        gd_zmin = zem > zmin
        keep = keep & gd_zmin #& gd_zlim
    #gd_zlim = (zem-dlasurvey.sightlines['Z_START']) > 0.1
    #pdb.set_trace()

    # Assess
    final = dlasurvey.sightlines[keep]
    sdict = {}
    sdict['n'] = len(final)
    print("We have {:d} sightlines for analysis".format(sdict['n']))

    def qck_stats(idict, tbl, istr, key):
        idict[istr+'min'] = np.min(tbl[key])
        idict[istr+'max'] = np.max(tbl[key])
        idict[istr+'median'] = np.median(tbl[key])
    qck_stats(sdict, final, 'z', 'ZEM')
    qck_stats(sdict, final, 'i', 'MAG')

    print("Min z = {:g}, Median z = {:g}, Max z = {:g}".format(sdict['zmin'], sdict['zmedian'], sdict['zmax']))

    # Return
    return final, sdict


def init_fNHI():
    """ Generate the interpolator for log NHI

    Returns
    -------
    fNHI : scipy.interpolate.interp1d function
    """
    from pyigm.fN.fnmodel import FNModel
    # f(N)
    fN_model = FNModel.default_model()
    # Integrate on NHI
    lX, cum_lX, lX_NHI = fN_model.calculate_lox(fN_model.zpivot,
                                                20.3, NHI_max=22.5, cumul=True)
    # Interpolator
    cum_lX /= cum_lX[-1] # Normalize
    fNHI = interpolate.interp1d(cum_lX, lX_NHI,
                                bounds_error=False,fill_value=lX_NHI[0])
    return fNHI


def insert_dlas(spec, zem, fNHI=None, rstate=None):
    """
    Parameters
    ----------
    spec
    fNHI
    rstate

    Returns
    -------
    final_spec : XSpectrum1D
    dlas : list
      List of DLAs inserted

    """
    from pyigm.fN import dla as pyi_fd
    from pyigm.abssys.dla import DLASystem
    from pyigm.abssys.utils import hi_model

    # Init
    if rstate is None:
        rstate = np.random.RandomState()
    if fNHI is None:
        fNHI = init_fNHI()

    # Allowed redshift placement
    ## Cut on zem and 910A rest-frame
    zlya = spec.wavelength.value/1215.67 - 1
    dz = np.roll(zlya,-1)-zlya
    dz[-1] = dz[-2]
    gdz = (zlya < zem) & (spec.wavelength > 910.*u.AA*(1+zem))

    # l(z)
    lz = pyi_fd.lX(zlya[gdz], extrap=True, calc_lz=True)
    cum_lz = np.cumsum(lz*dz[gdz])
    tot_lz = cum_lz[-1]
    fzdla = interpolate.interp1d(cum_lz/tot_lz, zlya[gdz],
                                 bounds_error=False,fill_value=np.min(zlya[gdz]))#

    # n DLA
    nDLA = 0
    while nDLA == 0:
        nval = rstate.poisson(tot_lz, 100)
        gdv = nval > 0
        if np.sum(gdv) == 0:
            continue
        else:
            nDLA = nval[np.where(gdv)[0][0]]

    # Generate DLAs
    dlas = []
    for jj in range(nDLA):
        # Random z
        zabs = float(fzdla(rstate.random_sample()))
        # Random NHI
        NHI = float(fNHI(rstate.random_sample()))
        dla = DLASystem((0.,0), zabs, (None,None), NHI)
        dlas.append(dla)

    # Insert
    vmodel, _ = hi_model(dlas, spec, fwhm=3.)
    # Add noise
    rand = rstate.randn(spec.npix)
    noise = rand * spec.sig * (1-vmodel.flux.value)
    final_spec = XSpectrum1D.from_tuple((vmodel.wavelength,
                                         spec.flux.value*vmodel.flux.value+noise,
                                         spec.sig))
    # Return
    return final_spec, dlas


def make_set(ntrain, slines, outroot=None, tol=1*u.arcsec, igmsp_survey='SDSS_DR7',
             frac_without=0., seed=1234, zmin=None, zmax=4.5):
    """ Generate a training set

    Parameters
    ----------
    ntrain : int
      Number of training sightlines to generate
    slines : Table
      Table of sightlines without DLAs (usually from SDSS or BOSS)
    igmsp_survey : str, optional
      Dataset name for spectra
    frac_without : float, optional
      Fraction of sightlines (on average) without a DLA
    seed : int, optional
    outroot : str, optional
      Root for output filenames
        root+'.fits' for spectra
        root+'.json' for DLA info
    zmin : float, optional
      Minimum redshift for training; defaults to min(slines['ZEM'])
    zmax : float, optional
      Maximum redshift to train on

    Returns
    -------

    """
    from linetools.spectra.utils import collate

    # Init and checks
    igmsp = IgmSpec()
    assert igmsp_survey in igmsp.groups
    rstate = np.random.RandomState(seed)
    rfrac = rstate.random_sample(ntrain)
    if zmin is None:
        zmin = np.min(slines['ZEM'])
    rzem = zmin + rstate.random_sample(ntrain)*(zmax-zmin)
    fNHI = init_fNHI()

    all_spec = []
    full_dict = {}
    # Begin looping
    for qq in range(ntrain):
        print("qq = {:d}".format(qq))
        full_dict[qq] = {}
        # Grab sightline
        isl = np.argmin(np.abs(slines['ZEM']-rzem[qq]))
        full_dict[qq]['sl'] = isl  # sightline
        specl, meta = igmsp.allspec_at_coord((slines['RA'][isl], slines['DEC'][isl]),
                                           groups=['SDSS_DR7'], tol=tol, verbose=False)
        assert len(specl) == 1
        spec = specl[0]
        # Meta data for header
        mdict = {}
        for key in meta[0].keys():
            mdict[key] = meta[0][key][0]
        mhead = Header(mdict)
        # Clear?
        if rfrac[qq] < frac_without:
            spec.meta['headers'][0] = mdict.copy() #mhead
            all_spec.append(spec)
            full_dict[qq]['nDLA'] = 0
            continue
        # Insert at least one DLA
        spec, dlas = insert_dlas(spec, mhead['zem'], rstate=rstate, fNHI=fNHI)
        spec.meta['headers'][0] = mdict.copy() #mhead
        all_spec.append(spec)
        full_dict[qq]['nDLA'] = len(dlas)
        for kk,dla in enumerate(dlas):
            full_dict[qq][kk] = {}
            full_dict[qq][kk]['NHI'] = dla.NHI
            full_dict[qq][kk]['zabs'] = dla.zabs

    # Generate one object
    final_spec = collate(all_spec)
    # Write?
    if outroot is not None:
        # Spectra
        final_spec.write_to_hdf5(outroot+'.hdf5')
        # Dict -> JSON
        gdict = ltu.jsonify(full_dict)
        ltu.savejson(outroot+'.json', gdict, overwrite=True, easy_to_read=True)
    # Return
    return final_spec, full_dict


def training_prod(seed, nruns, nsline, nproc=10, outpath='./'):
    """ Perform a full production run of training sightlines

    Parameters
    ----------
    seed
    nsline
    outpath

    Returns
    -------

    """
    from subprocess import Popen
    rstate = np.random.RandomState(seed)
    # Generate individual seeds
    seeds = np.round(100000*rstate.random_sample(nruns)).astype(int)

    # Start looping on processor
        # Loop on the systems
    nrun = -1
    while(nrun < nruns):
        proc = []
        for ss in range(nproc):
            nrun += 1
            if nrun == nruns:
                break
            # Run
            script = ['./scripts/dlaml_trainingset.py', str(seeds[nrun]), str(nsline), str(outpath)]
            proc.append(Popen(script))
        exit_codes = [p.wait() for p in proc]


def main(flg_tst, sdss=None, ml_survey=None):
    import os

    # Sightlines
    if (flg_tst % 2**1) >= 2**0:
        if sdss is None:
            sdss = DLASurvey.load_SDSS_DR5(sample='all')
        slines, sdict = grab_sightlines(sdss, flg_bal=0)

    # Test case of 100 sightlines
    if (flg_tst % 2**2) >= 2**1:
        # Make training set
        _, _ = make_set(100, slines, outroot='results/training_100')

    # Production runs
    if (flg_tst % 2**3) >= 2**2:
        #training_prod(123456, 5, 10, outpath=os.getenv('DROPBOX_DIR')+'/MachineLearning/DLAs/')  # TEST
        #training_prod(123456, 10, 500, outpath=os.getenv('DROPBOX_DIR')+'/MachineLearning/DLAs/')  # TEST
        training_prod(12345, 10, 5000, outpath=os.getenv('DROPBOX_DIR')+'/MachineLearning/DLAs/')  # TEST

# Test
if __name__ == '__main__':
    # Run from above src/
    #  I.e.   python src/training_set.py
    flg_tst = 0
    #flg_tst += 2**0   # Grab sightlines
    #flg_tst += 2**1   # First 100
    flg_tst += 2**2   # Production run of training - fixed

    main(flg_tst)
