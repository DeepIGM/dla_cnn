#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dla_cnn.desi.DesiMock import DesiMock
from dla_cnn.data_model.Sightline import Sightline
from os.path import join,exists
from matplotlib import pyplot as plt
import numpy as np
def load_DesiMocks(files,path):
    """
    Load each file in files as DesiMock, then return all of them as a list.
    --------------------------------------------------------------------
    prameters:
    
    files: list, the number of the fits files to be read
    path: the dir of the folder where the fits files are saved
    
    ----------------------------------------------------------------------
    return:
    
    Mocks: list, all files in DesiMock form 
    """
    Mocks = []
    Missing = ""
    for file in files:
        file_path = join(path,str(file))
        spectra = join(file_path,"spectra-16-%s.fits"%file)
        truth = join(file_path,"truth-16-%s.fits"%file)
        zbest = join(file_path,"zbest-16-%s.fits"%file)
        if exists(spectra) and exists(truth) and exists(zbest):
            mock = DesiMock()
            mock.read_fits_file(spectra,truth,zbest)
            Mocks.append(mock)
        else:
            Missing += str(file)+" "
    assert Missing=="", "Missing files:%s"%Missing +"Please check them!"
    return Mocks
def sightline_retriever(sightline,Mocks):
    """
    retrieve the sightline in a list of DesiMock using its id
    --------------------------------------------------------
    parameters:
    
    sightline:int, sightline id
    Mocks: list, the list of DesiMock
    
    --------------------------------------------------------
    return:
    
    sightline: dla_cnn.data_model.Sghtline.Sightline object, the sightline we need, if not find, return None
    """
    for mock in Mocks:
        if sightline in mock.data.keys():
            return mock.get_sightline(sightline, camera='b',rebin=True, normalize=True)
    return None

