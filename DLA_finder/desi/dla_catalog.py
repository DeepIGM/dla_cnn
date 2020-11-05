from astropy.table import Table
from astropy.io import fits
from pkg_resources import resource_filename
import numpy as np
def generate_qso_table(sightlines,pred_abs):
    """
    generate a QSO table for fitting BAO.
    
    Parameters
    ----------------
    sightlines: list of 'dla_cnn.data_model.Sightline.Sightline` object
    pred_abs:dict, the predicted absorbers for each sightline
    
    Return
    ----------------
    qso_tbl: astropy.Table object
    
    """
    a_plates=[]
    a_fibers=[]
    a_mjds=[]
    a_ra=[]
    a_dec=[]
    a_zqso=[]
    a_id=[]
    qso_tbl=Table()
    for ii in range(0,len(sightlines)):
        sightline=sightlines[ii]
        
        a_plates.append(sightline.id)
        a_fibers.append(sightline.id)
        a_mjds.append(sightline.id)
        a_ra.append(sightline.ra)
        a_dec.append(sightline.dec)
        a_zqso.append(sightline.z_qso)
        a_id.append(sightline.id)
    qso_tbl['Plate'] = a_plates
    qso_tbl['FiberID'] = a_fibers
    qso_tbl['MJD'] = a_mjds
    qso_tbl['TARGET_RA'] = a_ra
    qso_tbl['TARGET_DEC'] = a_dec
    qso_tbl['ZQSO'] = a_zqso
    qso_tbl['TARGETID'] = a_id
    qso_tbl.meta={'EXTNAME': 'QSOCAT'}
    return qso_tbl
def generate_dla_table(sightlines,pred_abs):
    """
    generate a DLA&sub-DLA table.
    
    Parameters
    ----------------
    sightlines: list of 'dla_cnn.data_model.Sightline.Sightline` object
    pred_abs:dict, the predicted absorbers for each sightline
    
    Return
    ----------------
    dla_tbl: astropy.Table object
    
    """
    a_plates=[]
    a_fibers=[]
    a_mjds=[]
    a_ra=[]
    a_dec=[]
    a_zqso=[]
    a_zdla=[]
    a_id=[]
    a_dlaid=[]
    a_nhi=[]
    a_conf=[]
    dla_tbl = Table()
    for ii in range(0,len(sightlines)):
        sightline=sightlines[ii]
        if pred_abs[ii]!=[]:
            for jj in range(0,len(pred_abs[ii])):
                pred_ab=pred_abs[ii][jj]
                a_plates.append(sightline.id)
                a_fibers.append(sightline.id)
                a_mjds.append(sightline.id)
                a_ra.append(sightline.ra)
                a_dec.append(sightline.dec)
                a_zqso.append(sightline.z_qso)
                a_id.append(sightline.id)
                a_zdla.append(pred_ab['z_dla'])
                a_nhi.append(pred_ab['column_density'])
                a_dlaid.append(str(sightline.id)+'00'+str(jj))
                a_conf.append(pred_ab['dla_confidence'])
    dla_tbl['TARGET_RA'] = a_ra
    dla_tbl['TARGET_DEC'] = a_dec
    dla_tbl['ZQSO'] = a_zqso
    dla_tbl['Z'] = a_zdla
    dla_tbl['TARGETID'] = a_id
    dla_tbl['DLAID'] = a_dlaid
    dla_tbl['NHI'] = a_nhi
    dla_tbl['DLA_CONFIDENCE']=a_conf
    dla_tbl.meta={'EXTNAME': 'DLACAT'}
    return dla_tbl
    
def catalog_fits(sightlines,pred_abs,dlafile=None,qsofile=None):
    """
    save DLA and QSO catalog.
    
    Parameters
    ----------------
    sightlines: list of 'dla_cnn.data_model.Sightline.Sightline` object
    pred_abs:dict, the predicted absorbers for each sightline
    dlafile: str
    qsofile: str
    
    Return
    ----------------
    None
    
    """
    dla_tbl=generate_dla_table(sightlines,pred_abs)
    #qso_tbl=generate_qso_table(sightlines,pred_abs)
    dla_tbl.write(dlafile,overwrite=True)
    qso_tbl.write(qsofile,overwrite=True)
