
import numpy as np 
from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.lists.linelist import LineList
from pyigm.abssys.dla import DLASystem
from pyigm.abssys.lls import LLSSystem
from pyigm.abssys.utils import hi_model
from dla_cnn.data_model.Dla import Dla
from dla_cnn.desi.preprocess import estimate_s2n
#generate uniform nhi
def uniform_NHI(slls=False, mix=False, high=False):#slls，mix，high 
    """ Generate uniform log NHI

    Returns
    -------
    NHI :float 
    """
    if slls:
        NHI=np.random.uniform(19.5,20.3)
    elif high:
        NHI=np.random.uniform(21.2,22.5)
    elif mix:
        NHI=np.random.uniform(19.3,22.5)
    else:
        NHI=np.random.uniform(20.3,22.5)
    return NHI

#generate zabs
def init_zabs(sightline):
    """ Generate uniform zabs
    Parameters
    ----------
    sightline:dla_cnn.data_model.Sigthline object
    
    Returns
    -------
    zabs :float 
    """
    lam = 10**sightline.loglam
    zlya=lam/1215.67 -1
    #3000km/s away from lya emission
    gdz = (lam < 0.99*1215.67*(1+sightline.z_qso)) & (lam> 912.*(1+sightline.z_qso))& (lam>= 3700)
    zabs=np.random.choice(list(zlya[gdz]))
    return zabs

#generate dla
def insert_dlas(sightline,nDLA, rstate=None, slls=False,
                mix=False, high=False, noise=False):
    """ Insert a DLA into input spectrum
    Also adjusts the noise
   
    Parameters
    ----------
    sightline
    nDLA：int
    rstate
    low_s2n : bool, optional
    noise: bool, optional
    Returns
    -------
    dlas : list
      List of DLAs inserted

    """
    #init
    if rstate is None:
        rstate = np.random.RandomState()
    #spec = XSpectrum1D.from_tuple((10**sightline.loglam,sightline.flux,sightline.sig))
    spec = XSpectrum1D.from_tuple((10**sightline.loglam,sightline.flux))#generate xspectrum1d odject
    # Generate DLAs
    dlas = []
    spec_dlas=[]
    for jj in range(nDLA):
        # Random z
        zabs = init_zabs(sightline)
        # Random NHI
        NHI = uniform_NHI(slls=slls, mix=mix, high=high)
        spec_dla = Dla((1+zabs)*1215.6701, NHI,'00'+str(jj))
        if (slls or mix):
            dla = LLSSystem((sightline.ra,sightline.dec), zabs, None, NHI=NHI)      
        else:
            dla = DLASystem((sightline.ra,sightline.dec), zabs, None, NHI)
        dlas.append(dla)
        spec_dlas.append(spec_dla)
    # Insert
    vmodel, _ = hi_model(dlas, spec, fwhm=3.)
    #add noise to voigt profile
    if noise:
        rand = rstate.randn(len(sightline.flux))   
        noise = rand * sightline.error * np.sqrt(1-vmodel.flux.value**2)
    else:
        noise=0
    #generate spec
    final_spec = XSpectrum1D.from_tuple((vmodel.wavelength,spec.flux.value*vmodel.flux.value+noise))
    #generate new sightline
    sightline.flux=final_spec.flux.value
    sightline.dlas=spec_dlas
    sightline.s2n=estimate_s2n(sightline)
    return dlas
    

