from astropy.io import fits
import numpy as np
import pdb
'''
Compare the CoSLA catalog with CNN DLA/SLLS Catalog 
'''


hdulist = fits.open('CoSLA.fits')
'''
open CoSLA candidate catalog
contains: Cai+ 16 five candidates
other 40 candidates picked in 2016A for Lick/Shane
All candidates are single sightline absorbers (do not reply on group environments).
All of them have Ly-beta covered from SDSS. 
'''

CoSLA_Lya = hdulist[1].data.wave
CoSLA_zabs = hdulist[1].data.zabs
CoSLA_plate= hdulist[1].data.plate
CoSLA_mjd= hdulist[1].data.mjd
CoSLA_fiber= hdulist[1].data.fiber
CoSLA_tau= hdulist[1].data.tau


CNN_hdu = fits.open('DR12_DLA_SLLS.fits')
'''
open CNN Catalog from
https://github.com/davidparks21/qso_lya_detection_pipeline/tree/master/dla_cnn/catalogs/boss_dr12
'''
CNN_NHI = CNN_hdu[1].data.NHI
CNN_plate= CNN_hdu[1].data.Plate
CNN_mjd= CNN_hdu[1].data.MJD
CNN_fiber= CNN_hdu[1].data.Fiber
CNN_zabs = CNN_hdu[1].data.zabs

'''
matching algorithm (currently slow)
'''
for i in range(0,len(CoSLA_Lya)):
    diff_plate= CNN_plate- CoSLA_plate[i]
    diff_mjd= CNN_mjd - CoSLA_mjd[i]
    diff_fiber= CNN_fiber- CoSLA_fiber[i]
    diff_zabs = CNN_zabs - CoSLA_zabs[i]
    kk= i
    for j in range(0,len(diff_zabs)):
        if (abs(diff_plate[j]) < 0.5 and abs(diff_mjd[j])<0.5 and \
            abs(diff_fiber[j])<0.5 and abs(diff_zabs[j])<0.02):
            '''
            if CNN and CoSLA have the same plate, mjd, fiber, and zabs, then they are matched
            '''
            statement= str(CoSLA_plate[i])+'_'+str(CoSLA_mjd[i])+'_'+str(CoSLA_fiber[i])+\
              ' find a CNN SLLS/DLA counterpart with N_HI= '+ str(CNN_NHI[j])
            print(statement)
        else:
            if (kk == i): 
                statement= str(CoSLA_plate[i])+'_'+str(CoSLA_mjd[i])+'_'+str(CoSLA_fiber[i])+\
                ' does not find CNN SLLS/DLA counterpart'
                print(statement)
                kk= kk+1
