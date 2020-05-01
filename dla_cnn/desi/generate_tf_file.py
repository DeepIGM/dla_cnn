import tensorflow as tf
from dla_cnn.desi.training_sets import split_sightline_into_samples,select_samples_50p_pos_neg
from dla_cnn.desi.preprocess import label_sightline
import numpy as np


def write_as_tf_file(sightlines, output_path):
    '''
    Generate fragment data as tfrecord files for training and testing
 
    Parameters
    ----------
    sightlines: dla_cnn.data_model.Sightline list
    output_path:str 
                '.../xxx.tfrecord'
    
    returns
    ----------
    tfrecord file: sightlineid,fragment flux,classification,offset,col_density,lam data
    '''
    #creat writer
    tf_writer = tf.io.TFRecordWriter(path=output_path, options=None)
    for sightline in sightlines:
        label_sightline(sightline)
        #get fragment data
        flux,classification,offsets,column_density,lam = split_sightline_into_samples(sightline)
        #Split 50/50 to have equal representation
        sample_masks=select_samples_50p_pos_neg(sightline)
        if len(sample_masks)!=0:
            feature_internal = {'sightlineid':tf.train.Feature(int64_list = tf.train.Int64List(value = [sightline.id])),\
                               'FLUX':tf.train.Feature(bytes_list = tf.train.BytesList(value = [flux[sample_masks,:].tostring()])),\
                               'labels_classifier':tf.train.Feature(bytes_list = tf.train.BytesList(value = [classification[sample_masks].tostring()])),\
                               'labels_offset':tf.train.Feature(bytes_list = tf.train.BytesList(value = [offsets[sample_masks].tostring()])),\
                               'col_density':tf.train.Feature(bytes_list = tf.train.BytesList(value = [column_density[sample_masks].tostring()])),\
                               'lam':tf.train.Feature(bytes_list = tf.train.BytesList(value = [lam[sample_masks,:].tostring()]))}
            feature_extern = tf.train.Features(feature=feature_internal)
            #creat example
            example = tf.train.Example(features = feature_extern)
            #Serialize example into strings
            example_str = example.SerializeToString()
            tf_writer.write(example_str)
    tf_writer.close()
    
def read_tf_file(input_path):
    '''
    read tfrecord file data
    Parameters
    ----------
    input_path:str
    
    Return
    ----------
    x:tensorflow.python.data.ops.dataset_ops.MapDataset object
    
    '''
    object_datasets= tf.data.TFRecordDataset(input_path)
    object_feature={
    'sightlineid':tf.io.FixedLenFeature([], tf.int64),
    'FLUX':tf.io.FixedLenFeature([], tf.string),
    'labels_classifier':tf.io.FixedLenFeature([], tf.string),
    'labels_offset':tf.io.FixedLenFeature([], tf.string),
    'col_density':tf.io.FixedLenFeature([], tf.string),
    'lam':tf.io.FixedLenFeature([], tf.string)
    }
    def _parse_function (exam_proto): 
        return tf.io.parse_single_example (exam_proto, object_feature)
    x = object_datasets.map(_parse_function)
    return x
    #extract data from x
    #for i in x:#every sightline
       #sightlineid=i['sightlineid'].numpy()
       #flux=np.fromstring(i['FLUX'].numpy(),dtype='float32')
       #kernel=400
       #flux.shape=(-1,kernel)#reshape flux
       #classification=np.fromstring(i['labels_classifier'].numpy(),dtype='float32')
       #offsets=np.fromstring(i['labels_offset'].numpy(),dtype='float32')
       #column_density=np.fromstring(i['col_density'].numpy(),dtype='float32')
       #lam=np.fromstring(i['lam'].numpy(),dtype='float32') 
       #lam.shape=(-1,kernel)#reshape lam
    #return sightlineid,flux,classification,offsets,column_density,lam
