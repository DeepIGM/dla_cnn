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
from dla_cnn.spectra_utils import get_lam_data
from dla_cnn.data_model.DataMarker import Marker
from scipy.interpolate import interp1d
from os.path import join, exists
from os import remove
import csv

# Set defined items
from dla_cnn.desi import defs
REST_RANGE = defs.REST_RANGE
kernel = defs.kernel


def label_sightline(sightline, kernel=kernel, REST_RANGE=REST_RANGE, pos_sample_kernel_percent=0.3):
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
    ix_dlas=[]
    coldensity_dlas=[]
    for dla in sightline.dlas:
        if (912<(dla.central_wavelength/(1+sightline.z_qso))<1220)&(dla.central_wavelength>=3700):
            ix_dlas.append(np.abs(lam[ix_dla_range]-dla.central_wavelength).argmin()) 
            coldensity_dlas.append(dla.col_density)    # column densities matching ix_dlas

    '''
    # FLUXES - Produce a 1748x400 matrix of flux values
    fluxes_matrix = np.vstack(map(lambda f,r:f[r-kernelrangepx:r+kernelrangepx],
                                  zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0])))
    '''

    # CLASSIFICATION (1 = positive sample, 0 = negative sample, -1 = border sample not used
    # Start with all samples zero
    classification = np.zeros((np.sum(ix_dla_range)), dtype=np.float32)
    # overlay samples that are too close to a known DLA, write these for all DLAs before overlaying positive sample 1's
    for ix_dla in ix_dlas:
        classification[ix_dla-samplerangepx*2:ix_dla+samplerangepx*2+1] = -1
        # Mark out Ly-B areas
        lyb_ix = sightline.get_lyb_index(ix_dla)
        classification[lyb_ix-samplerangepx:lyb_ix+samplerangepx+1] = -1
    # mark out bad samples from custom defined markers
    #for marker in sightline.data_markers:
        #assert marker.marker_type == Marker.IGNORE_FEATURE              # we assume there are no other marker types for now
        #ixloc = np.abs(lam_rest - marker.lam_rest_location).argmin()
        #classification[ixloc-samplerangepx:ixloc+samplerangepx+1] = -1
    # overlay samples that are positive
    for ix_dla in ix_dlas:
        classification[ix_dla-samplerangepx:ix_dla+samplerangepx+1] = 1

    # OFFSETS & COLUMN DENSITY
    offsets_array = np.full([np.sum(ix_dla_range)], np.nan, dtype=np.float32)     # Start all NaN markers
    column_density = np.full([np.sum(ix_dla_range)], np.nan, dtype=np.float32)
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
def rebin(sightline, v):
    """
    Resample and rebin the input Sightline object's data to a constant dlambda/lambda dispersion.
    Parameters
    ----------
    sightline: :class:`dla_cnn.data_model.Sightline.Sightline`
    v: float, and np.log(1+v/c) is dlambda/lambda, its unit is m/s, c is the velocity of light
    Returns
    -------
    :class:`dla_cnn.data_model.Sightline.Sightline`:
    """
    # TODO -- Add inline comments
    c = 2.9979246e8
    dlnlambda = np.log(1+v/c)
    wavelength = 10**sightline.loglam
    max_wavelength = wavelength[-1]
    min_wavelength = wavelength[0]
    pixels_number = int(np.round(np.log(max_wavelength/min_wavelength)/dlnlambda))+1
    new_wavelength = wavelength[0]*np.exp(dlnlambda*np.arange(pixels_number))
    
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
    
    left = 0
    while np.isnan(new_fx[left])|np.isnan(new_var[left]):
        left = left+1
    right = len(new_fx)
    while np.isnan(new_fx[right-1])|np.isnan(new_var[right-1]):
        right = right-1
    
    test = np.sum((np.isnan(new_fx[left:right]))|(np.isnan(new_var[left:right])))
    assert test==0, 'Missing value in this spectra!'
    
    sightline.loglam = np.log10(new_wavelength[left:right])
    sightline.flux = new_fx[left:right]
    sightline.error = new_var[left:right]
    
    return sightline


def normalize(sightline, full_wavelength, full_flux):
    """
    Normalize spectrum by dividing the mean value of continnum at lambda[left,right]
    ------------------------------------------
    parameters:
    
    sightline: dla_cnn.data_model.Sightline.Sightline object;
    camera : str, 'b' : the blue channel of the specctra, 'r': the r channel of the spectra,
                  'z' : the z channel of the spectra, 'all': all spectra
    
    --------------------------------------------
    return
    
    sightline: the sightline after normalized
    
    """
    blue_limit = 1420
    red_limit = 1480
    rest_wavelength = full_wavelength/(sightline.z_qso+1)
    assert blue_limit <= red_limit,"No Lymann-alpha forest, Please check this spectra: %i"%sightline.id#when no lymann alpha forest exists, assert error.
    #use the slice we chose above to normalize this spectra, normalize both flux and error array using the same factor to maintain the s/n.
    good_pix = (rest_wavelength>=blue_limit)&(rest_wavelength<=red_limit)
    sightline.flux = sightline.flux/np.median(full_flux[good_pix])
    sightline.error = sightline.error/np.median(full_flux[good_pix])
    
def estimate_s2n(sightline):
    """
    Estimate the s/n of a given sightline, using the lymann forest part and excluding dlas.
    -------------------------------------------------------------------------------------
    parameters；
    sightline: class:`dla_cnn.data_model.sightline.Sightline` object, we use it to estimate the s/n,
               and since we use the lymann forest part, the sightline's wavelength range should contain 1070~1170
    --------------------------------------------------------------------------------------
    return:
    s/n : float, the s/n of the given sightline.
    """
    #determine the lymann forest part of this sightline
    blue_limit = 1420
    red_limit = 1480
    wavelength = 10**sightline.loglam
    rest_wavelength = wavelength/(sightline.z_qso+1)
    #lymann forest part of this sightline, contain dlas 
    test = (rest_wavelength>blue_limit)&(rest_wavelength<red_limit)
    #when excluding the part of dla, we remove the part between central_wavelength+-delta
    #dwv = rest_wavelength[1]-rest_wavelength[0]#because we may change the re-sampling of the spectra, this need to be calculated.
    #dv = dwv/rest_wavelength[0] * 3e5  # km/s
    #delta = int(np.round(3000./dv))
    #for dla in sightline.dlas:
        #test = test&((wavelength>dla.central_wavelength+delta)|(wavelength<dla.central_wavelength-delta))
    #assert np.sum(test)>0, "this sightline doesn't contain lymann forest, sightline id: %i"%sightline.id
    s2n = sightline.flux/sightline.error
    #return s/n
    return np.median(s2n[test])

def generate_summary_table(sightlines, output_dir, mode = "w"):
    """
    Generate a csv file to store some necessary information of the given sightlines. The necessary information means the id, z_qso,
    s/n of thelymann forest part(avoid dlas+- 3000km/s), the wavelength range and corresponding pixel number of each channel.And the csv file's format is like:
    id(int)， z_qso(float), s2n(float), wavelength_start_b(float), wavelength_end_b(float),pixel_start_b(int), pixel_end_b(int), wavelength_start_r(float), wavelength_end_r(float),pixel_start_r(int), pixel_end_r(int), wavelength_start_z(float), wavelength_end_z(float),pixel_start_z(int), pixel_end_z(int),dlas_col_density(str),dlas_central_wavelength(str)
    "wavelength_start_b" means the start wavelength value of b channel, "wavelength_end_b" means the end wavelength value of b channel, "pixel_start_b" means the start pixel number of b channel,"pixel_end_b" means the end pixel number of b channel
    so do the other two channels.Besides, "dlas_col_density" means the col_density array of the sightline, and "dlas_central_wavelength" means the central wavelength array means the central wavelength array of the given sightline. Due to the uncertainty of the dlas' number, we chose to use str format to store the two arrays,
    each array is written in the format like "value1,value2, value3", and one can use `str.split(",")` to get the data, the column density and central wavelength which have the same index in the two arrayscorrspond to the same dla.
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    parameters:
    sightlines: list of `dla_cnn.data_model.Sightline.Sightline` object, the sightline contained should
    contain the all data of b,r,z channel, and shouldn't be rebinned,
    output_dir: str, where the output csv file is stored, its format should be "xxxx.csv",
    mode: str, possible values "w", "a", "w" means writing to the csv file directly(overwrite the 
    previous content), "a" means adding more data to the csv file(remaining the previous content)
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------
    return:
    None
    """
    #the header of the summary table, each element's meaning can refer to above comment
    headers = ["id","z_qso","s2n","wavelength_start_b","wavelength_end_b","pixel_start_b","pixel_end_b","wavelength_start_r","wavelength_end_r","pixel_start_r","pixel_end_r","wavelength_start_z","wavelength_end_z","pixel_start_z","pixel_end_z","dlas_col_density","dlas_central_wavelength"]
    #open the csv file
    with open(output_dir, mode=mode,newline="") as summary_table:
        summary_table_writer = csv.DictWriter(summary_table,headers)
        if mode == "w":
            summary_table_writer.writeheader()
        for sightline in sightlines:
            #for each sightline, read its information and write to the csv file
            info = {"id":sightline.id, "z_qso":sightline.z_qso, "s2n": sightline.s2n,"wavelength_start_b":10**sightline.loglam[0],
                    "wavelength_end_b":10**sightline.loglam[sightline.split_point_br-1],"pixel_start_b":0,"pixel_end_b":sightline.split_point_br-1,
                    "wavelength_start_r":10**sightline.loglam[sightline.split_point_br],"wavelength_end_r":10**sightline.loglam[sightline.split_point_rz-1],
                "pixel_start_r":sightline.split_point_br,"pixel_end_r":sightline.split_point_rz-1,"wavelength_start_z":10**sightline.loglam[sightline.split_point_rz],
                    "wavelength_end_z":10**sightline.loglam[-1],"pixel_start_z":sightline.split_point_rz,"pixel_end_z":len(sightline.loglam)-1}
            
            dlas_col_density = ""
            dlas_central_wavelength = ""
            for dla in sightline.dlas:
                dlas_col_density += str(dla.col_density)+","
                dlas_central_wavelength += str(dla.central_wavelength)+","
            info["dlas_col_density"] = dlas_col_density[:-1]
            info["dlas_central_wavelength"] = dlas_central_wavelength[:-1]
            
            #write to the csv file
            summary_table_writer.writerow(info)
#from dla_cnn.desi.DesiMock import DesiMock
def write_summary_table(nums, version,path, output_path):
    """
    Directly read data from fits files and write the summary table, the summary table contains all available sightlines(dlas!=[] and z_qso>2.33) in the given fits files.
    -----------------------------------------------------------------------------------------------------------------------------------------
    parameters:
    nums: list, the given fits files' id, its elements' format is int, and one should make sure all fits files are available before invoking this funciton, otherwise some sightlines can be missed;
    version: int, the version of the data set we use, e.g. if the version is v9.16, then version = 16
    path: str, the dir of the folder which stores the given fits file, the folder's structure is like folder-fits files' id - fits files , if you are still confused, you can check the below code about read data from the fits file;
    output_path: str, the dir where the summary table is generated, and if there have been a summary table, then we will remove it and generate a new summary table;
    ------------------------------------------------------------------------------------------------------------------------------------------
    retrun:
    None
    """
    #if exists summary table before, remove it
    #if exists(output_path):
        #remove(output_path)
    def write_as_summary_table(num):
        """
        write summary table for a single given fits file, if there have been a summary table then directly write after it, otherwise create a new one
        ---------------------------------------------------------------------------------------------------------------------------------------------
        parameter:
        num: int, the id of the given fits file, e.g. 700
        ---------------------------------------------------------------------------------------------------------------------------------------------
        return:
        None
        """
        #read data from fits file
        file_path = join(path,str(num))
        spectra = join(file_path,"spectra-%i-%i.fits"%(version,num))
        truth = join(file_path,"truth-%i-%i.fits"%(version,num))
        zbest = join(file_path,"zbest-%i-%i.fits"%(version,num))
        spec = DesiMock()
        spec.read_fits_file(spectra,truth,zbest)
        sightlines = []
        bad_sightlines = []
        for key in spec.data.keys():
            if spec.data[key]["z_qso"]>2.33 and spec.data[key]["DLAS"]!=[]:
                sightlines.append(spec.get_sightline(key))
        #generate summary table
        if exists(output_path):
            generate_summary_table(sightlines,output_path,"a")
        else:
            generate_summary_table(sightlines,output_path,"w")
    bad_files = [] #store the fits files with problems 
    #for each id in nums, invoking the `write_as_summary_table` funciton
    for num in nums:
        try:
            write_as_summary_table(num)
        except:
            #if have problems append to the bad_files
            bad_files.append(num)
    assert bad_files==[], "these fits files have some problems, check them please, fits files' id :%s"%str(bad_files)
