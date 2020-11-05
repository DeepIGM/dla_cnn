#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:40:41 2020
@author: samwang
"""

""" get the window prediction results and analyse"""
#!python

"""
0. Update all methods to TF 2.1 including using tf.Dataset
"""
import numpy as np
import math
import re, os, traceback, sys, json
import argparse
import tensorflow as tf
import timeit
from tensorflow.python.framework import ops
#from linetools.spectra.xspectrum1d import XSpectrum1D
#from pyigm.abssys.dla import DLASystem
#from pyigm.abssys.lls import LLSSystem
#from pyigm.abssys.utils import hi_model
#import matplotlib.pyplot as plt
ops.reset_default_graph()
#tf.compat.v1.disable_eager_execution()
#init = tf.compat.v1.global_variables_initializer()

from model import build_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
tensor_regex = re.compile('.*:\d*')
# Get a tensor by name, convenience method
def t(tensor_name):
    tensor_name = tensor_name+":0" if not tensor_regex.match(tensor_name) else tensor_name
    return tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
#change the window wavelength to the whole sightline wavelength
def generate_sightline(flux,lam):
    lam_sightline=[]
    flux_sightline=[]
    partsnumber=len(lam)#how many windows are included
    for k in range(0,partsnumber-1):
        lam_sightline.append(lam[k][0])
        flux_sightline.append(flux[k][0])
    for lams in lam[-1]:
        lam_sightline.append(lams)
    for fluxs in flux[-1]:
        flux_sightline.append(fluxs)
    lam_analyse=lam_sightline[199:-200]
    flux_analyse=flux_sightline[199:-200]
    return lam_analyse , flux_analyse #wavelength and flux for the whole sightline

def get_dla(zabs,NHI,matrix_lam,matrix_flux,wvoff=60.): #using XSpectrum1D in linetools to draw the DLA
    spec = XSpectrum1D.from_tuple((matrix_lam,matrix_flux))
    if NHI<20.3:
        dla = LLSSystem((0,0), zabs, None, NHI=NHI)      
    else:
        dla = DLASystem((0,0), zabs, None, NHI)
    wvcen = (1+zabs)*1215.67
    gd_wv = (spec.wavelength.value > wvcen-wvoff) & (spec.wavelength.value < wvcen+wvoff)
    co = np.amax(spec.flux[gd_wv])
    lya, lines = hi_model(dla, spec, lya_only=True)
    return lya.wavelength,lya.flux*co



def predictions_ann(hyperparameters, flux, checkpoint_filename, TF_DEVICE=''):
    timer = timeit.default_timer()
    BATCH_SIZE = 4000
    #n_samples = np.array(flux).shape[1]
    n_samples = flux.shape[0]
    pred = np.zeros((n_samples,), dtype=np.float32)
    conf = np.copy(pred)
    offset = np.copy(pred)
    coldensity = np.copy(pred) #set the 4 np matrix for every label


    with tf.Graph().as_default():
        build_model(hyperparameters)

        with tf.device(TF_DEVICE), tf.compat.v1.Session() as sess:
            tf.compat.v1.train.Saver().restore(sess, checkpoint_filename+".ckpt")
            for i in range(0,n_samples,BATCH_SIZE):
                pred[i:i+BATCH_SIZE], conf[i:i+BATCH_SIZE], offset[i:i+BATCH_SIZE], coldensity[i:i+BATCH_SIZE] = \
                    sess.run([t('prediction'), t('output_classifier'), t('y_nn_offset'), t('y_nn_coldensity')],
                             feed_dict={t('x'):                 flux[i:i+BATCH_SIZE,:],
                                        t('keep_prob'):         1.0})

    print("Localize Model processed {:d} samples in chunks of {:d} in {:0.1f} seconds".format(
          n_samples, BATCH_SIZE, timeit.default_timer() - timer))

    # coldensity_rescaled = coldensity * COL_DENSITY_STD + COL_DENSITY_MEAN
    return pred, conf, offset, coldensity #get the 4 labels prediction for every window


# Called from train_ann to perform a test of the train or test data, needs to separate pos/neg to get accurate #'s



if __name__ == '__main__':

    parameter_names = ["learning_rate", "training_iters", "batch_size", "l2_regularization_penalty", "dropout_keep_prob",
                       "fc1_n_neurons", "fc2_1_n_neurons", "fc2_2_n_neurons", "fc2_3_n_neurons",
                       "conv1_kernel", "conv2_kernel", "conv3_kernel",
                       "conv1_filters", "conv2_filters", "conv3_filters",
                       "conv1_stride", "conv2_stride", "conv3_stride",
                       "pool1_kernel", "pool2_kernel", "pool3_kernel",
                       "pool1_stride", "pool2_stride", "pool3_stride"]
    parameters = [
        # First column: Keeps the best best parameter based on accuracy score
        # Other columns contain the parameter options to try


        # learning_rate
        [0.000500,0.00002,         0.0005, 0.0007, 0.0010, 0.0030, 0.0050, 0.0070],
        # training_iters
        [100000],
        # batch_size
        [400,700,           400, 500, 600, 700, 850, 1000],
        # l2_regularization_penalty
        [0.005,         0.01, 0.008, 0.005, 0.003],
        # dropout_keep_prob
        [0.9,0.98,          0.75, 0.9, 0.95, 0.98, 1],
        # fc1_n_neurons
        [500,350,           200, 350, 500, 700, 900, 1500],
        # fc2_1_n_neurons
        [700,200,           200, 350, 500, 700, 900, 1500],
        # fc2_2_n_neurons
        [500,350,           200, 350, 500, 700, 900, 1500],
        # fc2_3_n_neurons
        [150,           200, 350, 500, 700, 900, 1500],
        # conv1_kernel
        [40,32,            20, 22, 24, 26, 28, 32, 40, 48, 54],
        # conv2_kernel
        [32,16,            10, 14, 16, 20, 24, 28, 32, 34],
        # conv3_kernel
        [20,16,            10, 14, 16, 20, 24, 28, 32, 34],
        # conv1_filters
        [100,           64, 80, 90, 100, 110, 120, 140, 160, 200],
        # conv2_filters
        [256,96,            80, 96, 128, 192, 256],
        # conv3_filters
        [128,96,            80, 96, 128, 192, 256],
        # conv1_stride
        [5,3,             2, 3, 4, 5, 6, 8],
        # conv2_stride
        [2,1,             1, 2, 3, 4, 5, 6],
        # conv3_stride
        [1,1,             1, 2, 3, 4, 5, 6],
        # pool1_kernel
        [7,             3, 4, 5, 6, 7, 8, 9],
        # pool2_kernel
        [4,6,             4, 5, 6, 7, 8, 9, 10],
        # pool3_kernel
        [6,             4, 5, 6, 7, 8, 9, 10],
        # pool1_stride
        [1,4,             1, 2, 4, 5, 6],
        # pool2_stride
        [5,4,             1, 2, 3, 4, 5, 6, 7, 8],
        # pool3_stride
        [6,4,             1, 2, 3, 4, 5, 6, 7, 8]
    ]
    

    # Random permutation of parameters out some artibrarily long distance
    #r = np.random.permutation(1000)

    # Write out CSV header
    


    #hyperparameters = [parameters[i][0] for i in range(len(parameters))]
    hyperparameters = {}
    for k in range(0,len(parameter_names)):
        hyperparameters[parameter_names[k]] = parameters[k][0]

    pred_dataset='/home/bwang/data/smoothtest/2461-snr3dataset.npy' #the prediction data file
    r=np.load(pred_dataset,allow_pickle = True,encoding='latin1').item()

    checkpoint_filename='/home/bwang/smoothtrain/train_921/current_99999' #model checkpoint_file (result from training)

    dataset={}

    delta_z=[]
    delta_NHI=[]

    TP=[]
    TN=[]
    FP=[]
    FN=[]
    #idlist=[]
    for sight_id in r.keys():
    
        flux=np.array(r[sight_id]['FLUXMATRIX'])
        #lam=r[sight_id]['lam']

        #(lam_analyse,flux_analyse)=generate_sightline(flux,lam)

        (pred, conf, offset, coldensity)=predictions_ann(hyperparameters, flux, checkpoint_filename, TF_DEVICE='')


        dataset[sight_id]={'pred':pred,'conf':conf,'offset': offset, 'coldensity':coldensity } #save the window prediction as npy file, use it to get sightline prediction later

        for p in range(0,len(pred)):
            if (r[sight_id]['labels_classifier'][p]==1) & (pred[p]==1):
                TP.append(p)
            if (r[sight_id]['labels_classifier'][p]==1) & (pred[p]==0):
                FN.append(p)
            if (r[sight_id]['labels_classifier'][p]==0) & (pred[p]==0):
                TN.append(p)
            if (r[sight_id]['labels_classifier'][p]==0) & (pred[p]==1):
                FP.append(p)


        NHI=[]
        for j in range(0,len(coldensity)):
            if pred[j]==0:
                NHI.append(0)
            else:
                NHI.append(coldensity[j])
        #set coldensity to 0 when there is no DLA in the window
        #print(len(coldensity))
        #print(NHI)
        #print(len(NHI))

    
        for p in range(0,len(NHI)):
            wave_dla=300+r[sight_id]['labels_offset'][p]
            wave_pre=300+offset[p]
            zabs=(r[sight_id]['lam'][p][int(wave_pre)]/1215.67)-1
            a_z=(r[sight_id]['lam'][p][int(wave_dla)]/1215.67)-1
            delta_z.append(zabs-a_z)
            #if (NHI[p]-r[sight_id]['col_density'][p] < 5) & (NHI[p]-r[sight_id]['col_density'][p] > -5):
            #if (pred[p]==1 ) & (r[sight_id]['labels_classifier'][p]==1):
            delta_NHI.append(NHI[p]-r[sight_id]['col_density'][p])


        print('%s is saved'%(sight_id))

    np.save('partpre/snrbin3/predbin3realdla.npy',dataset)

    import matplotlib.pyplot as plt
    print(delta_z)
    print(delta_NHI)
    arr_mean = np.mean(delta_z)
    arr_var = np.var(delta_z)
    arr_std = np.std(delta_z,ddof=1)
    arr_median=np.median(delta_z)
    print("average:%f" % arr_mean)
    print("fangcha:%f" % arr_var)
    print("biaozhuncha:%f" % arr_std)
    print('median:%f'%arr_median)

    arr_mean_2 = np.mean(delta_NHI)
    arr_var = np.var(delta_NHI)
    arr_std_2 = np.std(delta_NHI,ddof=1)
    arr_median_2=np.median(delta_NHI)
    print("average:%f" % arr_mean)
    print("fangcha:%f" % arr_var)
    print("biaozhuncha:%f" % arr_std_2)
    print('median:%f'%arr_median_2)

    plt.figure(figsize=(10,10))
    plt.title('stddev=%.3f median=%.3f'%(arr_std,arr_median),fontdict=None,loc='center',pad='20',fontsize=20,color='red')
    plt.hist(delta_z,bins=50,density=False,edgecolor='black')
    plt.ylabel('N',fontsize=20)
    plt.xlabel('$\Delta$'+'z',fontsize=20)
    plt.tick_params(labelsize=14)
    plt.savefig('partpre/snrbin3/delta_z-bin3totalrealdla.png')


    plt.figure(figsize=(10,10))
    plt.title('stddev=%.3f median=%.3f'%(arr_std_2,arr_mean_2),fontdict=None,loc='center',pad='20',fontsize=20,color='red')
    plt.hist(delta_NHI,bins=100,density=False,edgecolor='black')
    #plt.xlim(-3,3)
    #plt.ylim(0,10000)
    plt.ylabel('N',fontsize=20)
    plt.xlabel('$\Delta$'+'log${N_{\mathregular{HI}}}$',fontsize=20)
    plt.tick_params(labelsize=18)
    plt.savefig('partpre/snrbin3/delta_NHI-bin3realdla-total.png')


    print('samples of TP is %s'%(len(TP)))
    print('samples of TN is %s'%(len(TN)))
    print('samples of FP is %s'%(len(FP)))
    print('samples of FN is %s'%(len(FN)))