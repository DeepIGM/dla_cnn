import numpy as np 
from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.lists.linelist import LineList
from pyigm.abssys.dla import DLASystem
from pyigm.abssys.lls import LLSSystem
from pyigm.abssys.utils import hi_model
from dla_cnn.data_model.Dla import Dla
from dla_cnn.desi.preprocess import estimate_s2n
def uniform_NHI(slls=False, mix=False, high=False):
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

def init_zabs(sightline,overlap=False):
    """ Generate uniform zabs
    
    Parameters
    --------
    sightline: dla_cnn.data_model.sightline.Sightline object
    overlap: bool
    
    Returns
    -------
    zabs :float 
    """
    zindex=[]
    lam = 10**sightline.loglam
    zlya=lam/1215.67 -1
    #central wavelength for DLA:more than 3700 and 3000km/s away from lya emission
    gdz = (lam < 0.99*1215.67*(1+sightline.z_qso)) & (lam> 912.*(1+sightline.z_qso))& (lam>= 3700)
    index=np.array(range(0,len(zlya)))
    zindex.append(np.random.choice(list(index[gdz])))
    if overlap:
        #need to ensure
        gdz=gdz&((index<=zindex[0]-30)|(index>=zindex[0]+30))&(index<=zindex[0]+120)&(index>=zindex[0]-120)
        zindex.append(np.random.choice(list(index[gdz])))
    return zlya[zindex]
#generate DLA
def insert_dlas(sightline,overlap=False, rstate=None, slls=False,
                mix=False, high=False, noise=False):
    """ Insert a DLA into input spectrum
    Also adjusts the noise
    Will also add noise 'everywhere' if requested
    Parameters
    ----------
    sightline:dla_cnn.data_model.sightline.Sightline object
    overlap: bool
    noise: bool, optional
    
    Returns
    -------
    None

    """
    #init
    if rstate is None:
        rstate = np.random.RandomState()
    spec = XSpectrum1D.from_tuple((10**sightline.loglam,sightline.flux))#generate xspectrum1d
    # Generate DLAs
    dlas = []
    spec_dlas=[]
    zabslist = init_zabs(sightline,overlap)
    for zabs in zabslist:
        # Random NHI
        NHI = uniform_NHI(slls=slls, mix=mix, high=high)
        spec_dla = Dla((1+zabs)*1215.6701, NHI,'00'+str(jj))
        if (slls or mix):
            dla = LLSSystem((sightline.ra,sightline.dec), zabs, None, NHI=NHI)      
        else:
            dla = DLASystem((sightline.ra,sightline.dec), zabs, None, NHI)
        dlas.append(dla)
        spec_dlas.append(spec_dla)
    # Insert dlas to one sightline
    vmodel, _ = hi_model(dlas, spec, fwhm=3.)
    #add noise 
    if noise:
        rand = rstate.randn(len(sightline.flux))   
        noise = rand * sightline.error * np.sqrt(1-vmodel.flux.value**2)
    else:
        noise=0
    final_spec = XSpectrum1D.from_tuple((vmodel.wavelength,spec.flux.value*vmodel.flux.value+noise))
    #generate new sightline
    sightline.flux=final_spec.flux.value
    sightline.dlas=spec_dlas
    sightline.s2n=estimate_s2n(sightline)
    
    



