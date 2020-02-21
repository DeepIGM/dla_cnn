from astropy.io import fits
import numpy as np


class DesiMock:
    """
    a class to load all spectrum from a mock DESI data v9 fits file, each file contains 1186 spectra.
    :attribute wavelength array-like, the wavelength of all spectrum (all spectrum share same wavelength array)
    :attribute data, dict, using each spectra's id as its key and a dict of all data we need of this spectra as its value
      its format like spectra_id: {'FLUX':flux,'ERROR':error,'z_qso':z_qso, 'RA':ra, 'DEC':dec, 'DLAS':dlas information} ,
      dla information is a tuple, and its format is like (dla_id, z_qso, NHI)
    :attribute split_point_br, the length of the flux_b,the split point of  b channel data and r channel data
    :attribute split_point_rz, the length of the flux_b and flux_r, the split point of  r channel data and z channel data
    :attribute data_size, the point number of all data points of wavelength and flux
    """

    def _init_(self, wavelength = None, data = {}, split_point_br = None, split_point_rz = None, data_size = None):
        self.wavelength = wavelength
        self.data = data
        self.split_point_br = split_point_br
        self.split_point_rz = split_point_rz
        self.data_size = data_size

    def read_fits_file(self, spec_path, truth_path, zbest_path):
        """
        read Desi Mock spectrum from a fits file, load all spectrum as a DesiMock object
        :param spec_path: spectrum file path
        :param truth_path: truth file path
        :param zbest_path: zbest file path
        :return: self.wavelength,self.data(contained all information we need),self.split_point_br,self.split_point_rz,self.data_size
        """
        spec = fits.open(spec_path)
        truth = fits.open(truth_path)
        zbest = fits.open(zbest_path)

        #spec[2].data ,spec[7].data and spec[12].data are the wavelength data
        self.wavelength = np.hstack((spec[2].data.copy(), spec[7].data.copy(), spec[12].data.copy()))
        self.data_size = len(self.wavelength)

        dlas_data = truth[3].data[truth[3].data.copy()['NHI']>19.3]
        spec_dlas = {}
        #item[2] is the spec_id, item[3] is the dla_id, and item[0] is NHI, item[1] is z_qso
        for item in dlas_data:
            if item[2] not in spec_dlas:
                spec_dlas[item[2]] = [('00'+str(item[3]-item[2]*1000),item[1],item[0])]
            else:
                spec_dlas[item[2]].append(('00'+str(item[3]-item[2]*1000),item[1],item[0]))

        test = np.array([True if item in dlas_data['TARGETID'] else False for item in spec[1].data['TARGETID'].copy()])
        for item in spec[1].data['TARGETID'].copy()[~test]:
            spec_dlas[item] = []

        spec_id = spec[1].data['TARGETID'].copy()
        flux_b = spec[3].data.copy()
        flux_r = spec[8].data.copy()
        flux_z = spec[13].data.copy()
        flux = np.hstack((flux_b,flux_r,flux_z))
        ivar_b = spec[4].data.copy()
        ivar_r = spec[9].data.copy()
        ivar_z = spec[14].data.copy()
        error = 1./np.sqrt(np.hstack((ivar_b,ivar_r,ivar_z)))
        self.split_point_br = len(flux_b)
        self.split_point_rz = len(flux_b)+len(flux_z)
        z_qso = zbest[1].data['Z'].copy()
        ra = spec[1].data['TARGET_RA'].copy()
        dec = spec[1].data['TARGET_DEC'].copy()

        self.data = {spec_id[i]:{'FLUX':flux[i],'ERROR': error[i], 'z_qso':z_qso[i] , 'RA': ra[i], 'DEC':dec[i], 'DLAS':spec_dlas[spec_id[i]]} for i in range(len(spec_id))}

        return self.wavelength,self.data,self.split_point_br,self.split_point_rz,self.data_size

    def get_sightline(self,id):
        """
        using id(int) as index to retrive each spectra in DesiMock's dataset, return  a Sightline object.
        :param id: spectra's id , a unique number for each spectra
        :return sightline:
        """
        sightline = Sightline(id)
        sightline.flux = self.data[id]['FLUX']
        sightline.error = self.data[id]['ERROR']
        sightline.z_qso = self.data[id]['z_qso']
        sightline.ra = self.data[id]['RA']
        sightline.dec = self.data[id]['DEC']
        sightline.dlas = self.data[id]['DLAS']
        sightline.loglam = np.log10(self.wavelength)
        return sightline


