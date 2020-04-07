#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from dla_cnn.desi.load_fits_files import load_DesiMocks,sightline_retriever
from dla_cnn.desi.uniform_snr import split_bins


# In[2]:


def uniform_z_nhi(idarray,z_bins,nhi_bins,num,Mocks):
    """
    Uniform zdla,nhi,get sightline id
    -------------------------------------------------------------------------------
    parameters；
    idlist:numpy.ndarray
           sightlineid array after uniform snr
    z_bins:int
    nhi_bins:int
    num:int
    Mocks:list
          all files in DesiMock form 
    -------------------------------------------------------------------------------
    return:
    nhisample:list
    zsample:list
    sampleid:sightline idlist
    sample_dla_id:np.ndarray
    """
    
    nhi=[]
    zdla=[]
    dlaid=[]
    mockid=[]
    for ii in idlist:
        sightline=sightline_retriever(ii,Mocks)
        for jj in sightline.dlas:
            if 912<=(jj.central_wavelength/(1+sightline.z_qso))<=1220:#dla restframe[912,1220]
                nhi.append(jj.col_density) 
                zdla.append(float(jj.central_wavelength/1215.6701-1))
                mockid.append(ii)
                dlaid.append(str(ii)+jj.id)
    #cut NHI(20.3-21.5),zdla with rsd,dlaid
    nhicut=(np.array(nhi)>=20.3)&(np.array(nhi)<=21.2)&(np.array(zdla)>=2.13)#dla central wavelength>3800
    nhi_dla=np.array(nhi)[nhicut]
    zdla=np.array(zdla)[nhicut]
    mockid=np.array(mockid)[nhicut]
    dlaid=np.array(dlaid)[nhicut]
    #z split bins, get index
    z_index=split_bins(zdla,z_bins)
    #nhi split bins,get index
    nhi_index=[]
    for ii in range(0,z_bins):
        nhi_index.append(split_bins(nhi_dla[z_index[ii]],nhi_bins))
    #get bins id
    dlaid_bins=[]
    for i in range(0,z_bins):
        for j in range(0,nhi_bins):
            dlaid_bins.append(dlaid[z_index[i]][nhi_index[i][j]])
    #uniform idsample in every bins
    sample_dla_id=[]
    for jj in range(0,len(dlaid_bins)):
        sample_dla_id.append(np.random.choice(list(dlaid_bins[jj]),num))
    sample_dla_id=np.array(sample_dla_id).ravel()
    nhisample=[]
    zsample=[]
    sampleid=[]
    for i in range(0,len(sample_dla_id)): 
        index=np.where(dlaid==sample_dla_id[i])
        nhisample.append(nhi_dla[index])
        zsample.append(zdla[index])
        sampleid.append(mockid[index])


    return nhisample,zsample,sampleid,sample_dla_id








