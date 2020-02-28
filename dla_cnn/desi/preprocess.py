""" Code for pre-processing DESI data"""

''' Basic Recipe
0. Load the DESI mock spectrum
1. Resample to a constant dlambda/lambda dispersion
2. Renomalize the flux?
3. Generate a Sightline object with DLAs
4. Add labels 
5. Write to disk (numpy or TF)
'''

import numpy as np
from dla_cnn.desi.DesiMock import DesiMock
from dla_cnn.spectra_utils import get_lam_data
from dla_cnn.data_model.DataMarker import Marker

# Set defined items
#from dla_cnn.desi import defs
#REST_RANGE = defs.REST_RANGE
#kernel = defs.kernel


def label_sightline(sightline, kernel, REST_RANGE, pos_sample_kernel_percent=0.3):
    """
    Add labels to input sightline based on the DLAs along that sightline

    Parameters
    ----------
    sightline: dla_cnn.data_model.Sightline
    pos_sample_kernel_percent: float
    kernel: int
    REST_RANGE: list

    Returns
    -------
    classification: np.ndarray
        is 1 / 0 / -1 for DLA/nonDLA/border
    offsets_array: np.ndarray
        offset
    column_density: np.ndarray

    """
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    samplerangepx = int(kernel*pos_sample_kernel_percent/2) #60
    #kernelrangepx = int(kernel/2) #200
    ix_dlas = [(np.abs(lam[ix_dla_range]-dla.central_wavelength).argmin()) for dla in sightline.dlas]
    coldensity_dlas = [dla.col_density for dla in sightline.dlas]       # column densities matching ix_dlas

    '''
    # FLUXES - Produce a 1748x400 matrix of flux values
    fluxes_matrix = np.vstack(map(lambda f,r:f[r-kernelrangepx:r+kernelrangepx],
                                  zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0])))
    '''

    # CLASSIFICATION (1 = positive sample, 0 = negative sample, -1 = border sample not used
    # Start with all samples zero
    classification = np.zeros((REST_RANGE[2]), dtype=np.float32)
    # overlay samples that are too close to a known DLA, write these for all DLAs before overlaying positive sample 1's
    for ix_dla in ix_dlas:
        classification[ix_dla-samplerangepx*2:ix_dla+samplerangepx*2+1] = -1
        # Mark out Ly-B areas
        lyb_ix = sightline.get_lyb_index(ix_dla)
        classification[lyb_ix-samplerangepx:lyb_ix+samplerangepx+1] = -1
    # mark out bad samples from custom defined markers
    for marker in sightline.data_markers:
        assert marker.marker_type == Marker.IGNORE_FEATURE              # we assume there are no other marker types for now
        ixloc = np.abs(lam_rest - marker.lam_rest_location).argmin()
        classification[ixloc-samplerangepx:ixloc+samplerangepx+1] = -1
    # overlay samples that are positive
    for ix_dla in ix_dlas:
        classification[ix_dla-samplerangepx:ix_dla+samplerangepx+1] = 1

    # OFFSETS & COLUMN DENSITY
    offsets_array = np.full([REST_RANGE[2]], np.nan, dtype=np.float32)     # Start all NaN markers
    column_density = np.full([REST_RANGE[2]], np.nan, dtype=np.float32)
    # Add DLAs, this loop will work from the DLA outward updating the offset values and not update it
    # if it would overwrite something set by another nearby DLA
    for i in range(int(samplerangepx+1)):
        for ix_dla,j in zip(ix_dlas,range(len(ix_dlas))):
            offsets_array[ix_dla+i] = -i if np.isnan(offsets_array[ix_dla+i]) else offsets_array[ix_dla+i]
            offsets_array[ix_dla-i] =  i if np.isnan(offsets_array[ix_dla-i]) else offsets_array[ix_dla-i]
            column_density[ix_dla+i] = coldensity_dlas[j] if np.isnan(column_density[ix_dla+i]) else column_density[ix_dla+i]
            column_density[ix_dla-i] = coldensity_dlas[j] if np.isnan(column_density[ix_dla-i]) else column_density[ix_dla-i]
    offsets_array = np.nan_to_num(offsets_array)
    column_density = np.nan_to_num(column_density)

    # Append these to the Sightline
    sightline.classification = classification
    sightline.offsets = offsets_array
    sightline.column_density = column_density

    # classification is 1 / 0 / -1 for DLA/nonDLA/border
    # offsets_array is offset
    return classification, offsets_array, column_density

def rebin(sightline, v = 20000):
    """
    Resample and rebin the input Sightline object's data to a constant dlambda/lambda dispersion.

    Parameters
    ----------
    sightline: dla_cnn.data_model.Sightline
    v: float, and np.log(1+v/c) is dlambda/lambda
    
    Returns
    -------
    sightline: dla_cnn.data_model.Sightline
    """
    c = 2.9979246e8
    dlambda = np.log(1+v/c)
    wavelength = 10**sightline.loglam
    max_wavelength = wavelength[-1]
    min_wavelength = wavelength[0]
    pixels_number = int(np.round(np.log(max_wavelength/min_wavelength)/dlambda))+1
    new_wavelength = wavelength[0]*np.exp(dlambda*np.arange(pixels_number))
    
    npix = len(wavelength)
    wvh = (wavelength + np.roll(wavelength, -1)) / 2.
    wvh[npix - 1] = wavelength[npix - 1] + \
                    (wavelength[npix - 1] - wavelength[npix - 2]) / 2.
    dwv = wvh - np.roll(wvh, 1)
    dwv[0] = 2 * (wvh[0] - wavelength[0])
    med_dwv = np.median(dwv)
    
    cumsum = np.cumsum(sightline.flux * dwv)
    cumvar = np.cumsum(sightline.error * dwv, dtype=np.float64)
    
    fcum = interp1d(wvh, cumsum,bounds_error=False)
    fvar = interp1d(wvh, cumvar,bounds_error=False)
    
    nnew = len(new_wavelength)
    nwvh = (new_wavelength + np.roll(new_wavelength, -1)) / 2.
    nwvh[nnew - 1] = new_wavelength[nnew - 1] + \
                     (new_wavelength[nnew - 1] - new_wavelength[nnew - 2]) / 2.
    
    bwv = np.zeros(nnew + 1)
    bwv[0] = new_wavelength[0] - (new_wavelength[1] - new_wavelength[0]) / 2.
    bwv[1:] = nwvh
    
    newcum = fcum(bwv)
    newvar = fvar(bwv)
    
    new_fx = (np.roll(newcum, -1) - newcum)[:-1]
    new_var = (np.roll(newvar, -1) - newvar)[:-1]
    
    # Normalize (preserve counts and flambda)
    new_dwv = bwv - np.roll(bwv, 1)
    new_fx = new_fx / new_dwv[1:]
    # Preserve S/N (crudely)
    med_newdwv = np.median(new_dwv)
    new_var = new_var / (med_newdwv/med_dwv) / new_dwv[1:]
    
    test = (~np.isnan(new_fx))|(~np.isnan(new_var))
    sightline.loglam = np.log10(new_wavelength[test])
    sightline.flux = new_fx[test]
    sightline.error = new_var[test]
    
    return sightline
