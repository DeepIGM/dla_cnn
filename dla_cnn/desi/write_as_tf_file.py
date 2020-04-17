from dla_cnn.desi.training_sets import split_sightline_into_samples,select_samples_50p_pos_neg
from dla_cnn.desi.preprocess import label_sightline
import tensorflow as tf

def write_as_tf_file(sightline, output_path):
    '''
    sightline:dla_cnn.data_model.Sightline
    
    Returns:
    '''

    label_sightline(sightline, kernel=400, REST_RANGE=[900,1346], pos_sample_kernel_percent=0.3)
    flux,classificiation,offsets,column_density = split_sightline_into_samples(sightline, REST_RANGE=[900,1346], kernel=400)
    sample_masks=select_samples_50p_pos_neg(sightline,kernel=400)
    if len(sample_masks)!=0:
        tf_writer = tf.python_io.TFRecordWriter(path=join(output_path,"%i.tfrecord"%sightline.id), options=None)
        feature_internal = {"FLUX":tf.train.Feature(bytes_list = tf.train.BytesList(value = [flux[sample_masks,:].tostring()])),\
                            "labels_classifier":tf.train.Feature(bytes_list = tf.train.BytesList(value = [classificiation[sample_masks].tostring()])),\
                           "labels_offset":tf.train.Feature(bytes_list = tf.train.BytesList(value = [offsets[sample_masks].tostring()])),\
                           "col_density":tf.train.Feature(bytes_list = tf.train.BytesList(value = [column_density[sample_masks].tostring()]))}
        feature_extern = tf.train.Features(feature=feature_internal)
        example = tf.train.Example(features = feature_extern)
        example_str = example.SerializeToString()
        tf_writer.write(example_str)
        tf_writer.close()
