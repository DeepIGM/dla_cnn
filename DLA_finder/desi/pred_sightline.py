import numpy as np 
from dla_cnn.data_model.Sightline import Sightline
from dla_cnn.data_model.Prediction import Prediction
from dla_cnn.spectra_utils import get_lam_data
from dla_cnn.desi.analyze_prediction import analyze_pred

def save_pred(sightlines,pred,PEAK_THRESH,level):
    """
    convert  the parts prediction into whole sightline prediction.
    
    Parameters
    ------------------
    sightlines: list of dla_cnn.data_model.Sightline object
    pred:dict, model's output
    PEAK_THRESH:float
    level:float, if the conf of every pixel> level, the classification=1
     
    Return
    ------------------
    pred_abs:list of absorbers for given sightlines
    
    """
    pred_abs=[]
    #get absorbers list for each sightline
    for ii in range(0,len(sightlines)):
        sightline=sightlines[ii]
        conf=pred[sightline.id]['conf']
        #using different conf level to change classifier value
        #classifier=[]
        #for ii in range(0,len(conf)):
            #if conf[ii]>level:
                #classifier.append(1)
            #else:
                #classifier.append(0)
        #classifier=np.array(classifier)
        #just use the given classifier
        classifier=pred[sightline.id]['pred']
        offset=pred[sightline.id]['offset']
        coldensity=pred[sightline.id]['coldensity']
        pred_abs.append(analyze_pred(sightline,classifier,conf,offset,coldensity,PEAK_THRESH))
    return pred_abs

def get_results(sightlines,pred):
    """
    compare real absorbers and predicted absorbers and calculate TP,FN,FP, draw hist.
    
    Parameters
    ---------------
    sightlines:list of dla_cnn.data_model.Sightline object
    pred: list of absorbers for given sightlines, the output of save_pred module.
    
    Return
    ---------------
    tp,fn,fp:the index of TP,FN,FP sightlines
    fn_pred: the missing absorber's data: central wavelength and colomn density
    fp_pred: the FP absorber's data:central wavelength and colomn density
    
    """
    tp=[]
    fp=[]
    fn=[]
    fp_pred=[]
    tp_pred=[]
    fn_pred=[]
    for ii in range(0,len(sightlines)):
        sightline=sightlines[ii]
        pred_abs=pred[ii]
      
        central_wave=[]#pred wavelength
        col_density=[]#pred nhi
        real_wave=[]#real value
        real_density=[]#real value      
        if pred_abs!=[]:
            for pred_ab in pred_abs:#get pred list
                if (pred_ab['type']!='LYB'):#exclude lyb 
                        central_wave.append(pred_ab['spectrum'])
                        col_density.append(pred_ab['column_density'])
                        
            #find the tp,fn prediction corresponding to each real absorber
            for dla in sightline.dlas:#get real list
                #if (dla.central_wavelen:gth>=3800)&(dla.col_density>=20.0):
                if dla.col_density>=19.3:
                    real_wave.append(dla.central_wavelength)
                    real_density.append(dla.col_density)
                    lam_difference= np.abs(central_wave - dla.central_wavelength)
                    nearest_ix = np.argmin(lam_difference)
                    if lam_difference[nearest_ix]<=10:#delta lam<=10
                        tp.append(ii)
                        tp_pred.append([dla.central_wavelength,dla.col_density,central_wave[nearest_ix],col_density[nearest_ix]])   
                    else:#missing absorber
                        fn.append(ii)
                        fn_pred.append([dla.central_wavelength,dla.col_density])
                        
            #find the wrong prediction without corresponding to each real absorber          
            for i in range(0,len(central_wave)):
                if real_wave!=[]:
                     wave_difference=np.abs(np.array(real_wave) - central_wave[i])
                     nearestix = np.argmin(wave_difference)
                     if (wave_difference[nearestix]>15)&(col_density[i]>=19.3):#&(central_wave[i]<=1216*(1+sightline.z_qso)):
                        fp.append(ii)
                        fp_pred.append([central_wave[i],col_density[i]])
                else:#sightline without absorbers
                    fp.append(ii)
                    fp_pred.append([central_wave[i],col_density[i]])
                    
        else:
            for dla in sightline.dlas:
                #if (dla.central_wavelength>=3800)&(dla.col_density>=20.0):
                if dla.col_density>19.3:
                    fn.append(ii)
                    fn_pred.append([dla.central_wavelength,dla.col_density])
                      
    #draw NHI and z hist
    delta_z=[]
    delta_NHI=[]
    for pred in tp_pred:
        pred_z=pred[2]/1215.67-1
        real_z=pred[0]/1215.67-1
        delta_z.append(pred_z-real_z)
        delta_NHI.append(pred[3]-pred[1])
    #calculate mean,std,variance
    arr_mean = np.mean(delta_z)
    arr_var = np.var(delta_z)
    arr_std = np.std(delta_z,ddof=1)
    print("average:%f" % arr_mean)
    print("fangcha:%f" % arr_var)
    print("biaozhuncha:%f" % arr_std)
    arr_mean_2 = np.mean(delta_NHI)
    arr_var_2 = np.var(delta_NHI)
    arr_std_2 = np.std(delta_NHI,ddof=1)
    print("average:%f" % arr_mean_2)
    print("fangcha:%f" % arr_var_2)
    print("biaozhuncha:%f" % arr_std_2)
    
    plt.figure(figsize=(5,5))
    plt.title('stddev=%.4f mean=%.5f'%(arr_std,arr_mean),fontdict=None,loc='center',pad='20',fontsize=20,color='red')
    plt.hist(delta_z,bins=50,density=False,edgecolor='black')
    plt.ylabel('N',fontsize=20)
    plt.xlabel('$\Delta$'+'z',fontsize=20)
    plt.tick_params(labelsize=18)
    #plt.savefig('/Users/zjq/sightline/717/bin4delta_z.png')

    plt.figure(figsize=(5,5))
    plt.title('stddev=%.4f mean=%.5f'%(arr_std_2,arr_mean_2),fontdict=None,loc='center',pad='20',fontsize=20,color='red')
    plt.hist(delta_NHI,bins=100,density=False,edgecolor='black')
    plt.ylabel('N',fontsize=20)
    plt.xlabel('$\Delta$'+'log${N_{\mathregular{HI}}}$',fontsize=20)
    plt.tick_params(labelsize=18)
    #plt.savefig('/Users/zjq/sightline/717/bin4delta_NHI.png')
     
    return tp,fn,fp,fn_pred,fp_pred
