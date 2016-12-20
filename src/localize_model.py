#!python


import tensorflow as tf
import numpy as np
import random, os, sys, traceback, math, json, timeit, gc, multiprocessing, gzip, pickle, peakutils, re, scipy, getopt, argparse
from Dataset import Dataset
from scipy.signal import find_peaks_cwt
import scipy.signal as signal

# Mean and std deviation of distribution of column densities
# COL_DENSITY_MEAN = 20.488289796394628  
# COL_DENSITY_STD = 0.31015769579662766
tensor_regex = re.compile('.*:\d*')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv1d(x, W, s):
    return tf.nn.conv2d(x, W, strides=s, padding='SAME')


def pooling_layer_parameterized(pool_method, h_conv, pool_kernel, pool_stride):
    if pool_method == 1:
        return tf.nn.max_pool(h_conv, ksize=[1, pool_kernel, 1, 1], strides=[1, pool_stride, 1, 1], padding='SAME')
    elif pool_method == 2:
        return tf.nn.avg_pool(h_conv, ksize=[1, pool_kernel, 1, 1], strides=[1, pool_stride, 1, 1], padding='SAME')


def variable_summaries(var, name, collection):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries') as r:
        mean = tf.reduce_mean(var)
        tf.add_to_collection(collection, tf.scalar_summary('mean/' + name, mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.add_to_collection(collection, tf.scalar_summary('stddev/' + name, stddev))
        tf.add_to_collection(collection, tf.scalar_summary('max/' + name, tf.reduce_max(var)))
        tf.add_to_collection(collection, tf.scalar_summary('min/' + name, tf.reduce_min(var)))
        tf.add_to_collection(collection, tf.histogram_summary(name, var))


def build_model(hyperparameters):
    learning_rate = hyperparameters['learning_rate']
    l2_regularization_penalty = hyperparameters['l2_regularization_penalty']
    fc1_n_neurons = hyperparameters['fc1_n_neurons']
    fc2_1_n_neurons = hyperparameters['fc2_1_n_neurons']
    fc2_2_n_neurons = hyperparameters['fc2_2_n_neurons']
    fc2_3_n_neurons = hyperparameters['fc2_3_n_neurons']
    conv1_kernel = hyperparameters['conv1_kernel']
    conv2_kernel = hyperparameters['conv2_kernel']
    conv1_filters = hyperparameters['conv1_filters']
    conv2_filters = hyperparameters['conv2_filters']
    conv1_stride = hyperparameters['conv1_stride']
    conv2_stride = hyperparameters['conv2_stride']
    pool1_kernel = hyperparameters['pool1_kernel']
    pool2_kernel = hyperparameters['pool2_kernel']
    pool1_stride = hyperparameters['pool1_stride']
    pool2_stride = hyperparameters['pool2_stride']
    pool1_method = hyperparameters['pool1_method']
    pool2_method = hyperparameters['pool2_method']

    INPUT_SIZE = 400
    tfo = {}    # Tensorflow objects

    x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='x')
    label_classifier = tf.placeholder(tf.float32, shape=[None], name='label_classifier')
    label_offset = tf.placeholder(tf.float32, shape=[None], name='label_offset')
    label_coldensity = tf.placeholder(tf.float32, shape=[None], name='label_coldensity')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # First Convolutional Layer
    # Kernel size (16,1)
    # Stride (4,1)
    # number of filters = 4 (features?)
    # Neuron activation = ReLU (rectified linear unit)

    W_conv1 = weight_variable([conv1_kernel, 1, 1, conv1_filters])
    b_conv1 = bias_variable([conv1_filters])
    x_4d = tf.reshape(x, [-1, INPUT_SIZE, 1, 1])

    # https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#convolution
    # out_height = ceil(float(in_height) / float(strides[1])) = ceil(1024./4.) = 256
    # out_width = ceil(float(in_width) / float(strides[2])) = 1
    # shape of h_conv1: [-1, 256, 1, 4]
    stride1 = [1, conv1_stride, 1, 1]
    h_conv1 = tf.nn.relu(conv1d(x_4d, W_conv1, stride1) + b_conv1)

    # Kernel size (8,1)
    # Stride (2,1)
    # Pooling type = Max Pooling
    # out_height = ceil(float(in_height) / float(strides[1])) = ceil(256./2.) = 128
    # out_width = ceil(float(in_width) / float(strides[2])) = 1
    # shape of h_pool1: [-1, 128, 1, 4]
    h_pool1 = pooling_layer_parameterized(pool1_method, h_conv1, pool1_kernel, pool1_stride)

    # Second Convolutional Layer
    # Kernel size (16,1)
    # Stride (2,1)
    # number of filters=8
    # Neuron activation = ReLU (rectified linear unit)
    W_conv2 = weight_variable([conv2_kernel, 1, conv1_filters, conv2_filters])
    b_conv2 = bias_variable([conv2_filters])
    # out_height = ceil(float(in_height) / float(strides[1])) = ceil(128./2.) = 64
    # out_width = ceil(float(in_width) / float(strides[2])) = 1
    # shape of h_conv1: [-1, 64, 1, 8]
    stride2 = [1, conv2_stride, 1, 1]
    h_conv2 = tf.nn.relu(conv1d(h_pool1, W_conv2, stride2) + b_conv2)
    h_pool2 = pooling_layer_parameterized(pool2_method, h_conv2, pool2_kernel, pool2_stride)

    # FC1: first fully connected layer, shared
    inputsize_fc1 = int(math.ceil(math.ceil(math.ceil(math.ceil(
        INPUT_SIZE / conv1_stride) / pool1_stride) / conv2_stride) / pool2_stride)) * conv2_filters
    h_pool2_flat = tf.reshape(h_pool2, [-1, inputsize_fc1])

    W_fc1 = weight_variable([inputsize_fc1, fc1_n_neurons])
    b_fc1 = bias_variable([fc1_n_neurons])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout FC1
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2_1 = weight_variable([fc1_n_neurons, fc2_1_n_neurons])
    b_fc2_1 = bias_variable([fc2_1_n_neurons])
    W_fc2_2 = weight_variable([fc1_n_neurons, fc2_2_n_neurons])
    b_fc2_2 = bias_variable([fc2_2_n_neurons])
    W_fc2_3 = weight_variable([fc1_n_neurons, fc2_3_n_neurons])
    b_fc2_3 = bias_variable([fc2_3_n_neurons])

    # FC2 activations
    h_fc2_1 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2_1) + b_fc2_1)
    h_fc2_2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2_2) + b_fc2_2)
    h_fc2_3 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2_3) + b_fc2_3)

    # FC2 Dropout [1-3]
    h_fc2_1_drop = tf.nn.dropout(h_fc2_1, keep_prob)
    h_fc2_2_drop = tf.nn.dropout(h_fc2_2, keep_prob)
    h_fc2_3_drop = tf.nn.dropout(h_fc2_3, keep_prob)

    # Readout Layer
    W_fc3_1 = weight_variable([fc2_1_n_neurons, 1])
    b_fc3_1 = bias_variable([1])
    W_fc3_2 = weight_variable([fc2_2_n_neurons, 1])
    b_fc3_2 = bias_variable([1])
    W_fc3_3 = weight_variable([fc2_3_n_neurons, 1])
    b_fc3_3 = bias_variable([1])

    # y_fc4 = tf.add(tf.matmul(h_fc3_drop, W_fc4), b_fc4)
    # y_nn = tf.reshape(y_fc4, [-1])
    y_fc4_1 = tf.add(tf.matmul(h_fc2_1_drop, W_fc3_1), b_fc3_1)
    y_nn_classifier = tf.reshape(y_fc4_1, [-1], name='y_nn_classifer')
    y_fc4_2 = tf.add(tf.matmul(h_fc2_2_drop, W_fc3_2), b_fc3_2)
    y_nn_offset = tf.reshape(y_fc4_2, [-1], name='y_nn_offset')
    y_fc4_3 = tf.add(tf.matmul(h_fc2_3_drop, W_fc3_3), b_fc3_3)
    y_nn_coldensity = tf.reshape(y_fc4_3, [-1], name='y_nn_coldensity')

    # Train and Evaluate the model
    loss_classifier = tf.add(tf.nn.sigmoid_cross_entropy_with_logits(y_nn_classifier, label_classifier),
                             l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
                                                          tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2_1)),
                             name='loss_classifier')
    loss_offset_regression = tf.add(tf.reduce_sum(tf.nn.l2_loss(y_nn_offset - label_offset)),
                                    l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
                                                                 tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2_2)),
                                    name='loss_offset_regression')
    # loss_coldensity_regression = tf.add(tf.reduce_sum(tf.nn.l2_loss(y_nn_coldensity - label_coldensity)),
    #                                     l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
    #                                                                  tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2_3)),
    #                                     name='loss_coldensity_regression')
    # L = (y - y_hat) * (y / (y+1e-6)) + L2 regularization
    # The second term is 0 when the label is 0 and ~1 when the label is non zero
    epsilon = 1e-6#np.nextafter(0,1,dtype=np.float32)
    loss_coldensity_regression = tf.reduce_sum(
        tf.mul(tf.square(y_nn_coldensity - label_coldensity),
               tf.div(label_coldensity,label_coldensity+epsilon)) +
        l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
                                     tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2_1)),
       name='loss_coldensity_regression')

    # if hyperparameters['optimizer'] == 'ADAM':
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # elif hyperparameters['optimizer'] == 'ADADELTA':
    #     optimizer = tf.train.AdadeltaOptimizer()

    cost_all_samples_lossfns_AB = loss_classifier + loss_offset_regression
    cost_pos_samples_lossfns_ABC = loss_classifier + loss_offset_regression + loss_coldensity_regression
    # train_step_AB = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all_samples_lossfns_AB, global_step=global_step, name='train_step_AB')
    train_step_ABC = optimizer.minimize(cost_pos_samples_lossfns_ABC, global_step=global_step, name='train_step_ABC')
    # train_step_C = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_coldensity_regression, global_step=global_step, name='train_step_C')
    output_classifier = tf.sigmoid(y_nn_classifier, name='output_classifier')
    prediction = tf.round(output_classifier, name='prediction')
    correct_prediction = tf.equal(prediction, label_classifier)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    rmse_offset = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y_nn_offset,label_offset))), name='rmse_offset')
    rmse_coldensity = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y_nn_coldensity,label_coldensity))), name='rmse_coldensity')

    variable_summaries(loss_classifier, 'loss_classifier', 'SUMMARY_A')
    variable_summaries(loss_offset_regression, 'loss_offset_regression', 'SUMMARY_B')
    variable_summaries(loss_coldensity_regression, 'loss_coldensity_regression', 'SUMMARY_C')
    variable_summaries(accuracy, 'classification_accuracy', 'SUMMARY_A')
    variable_summaries(rmse_offset, 'rmse_offset', 'SUMMARY_B')
    variable_summaries(rmse_coldensity, 'rmse_coldensity', 'SUMMARY_C')
    # tb_summaries = tf.merge_all_summaries()

    return train_step_ABC, tfo #, accuracy , loss_classifier, loss_offset_regression, loss_coldensity_regression, \
           #x, label_classifier, label_offset, label_coldensity, keep_prob, prediction, output_classifier, y_nn_offset, \
           #rmse_offset, y_nn_coldensity, rmse_coldensity


# def predictions_ann_multiprocess(param_tuple):
#     return predictions_ann(param_tuple[0], param_tuple[1], param_tuple[2], param_tuple[3])


def predictions_ann(hyperparameters, flux, checkpoint_filename, TF_DEVICE=''):
    timer = timeit.default_timer()
    BATCH_SIZE = 4000
    n_samples = flux.shape[0]
    pred = np.zeros((n_samples,), dtype=np.float32)
    conf = np.copy(pred)
    offset = np.copy(pred)
    coldensity = np.copy(pred)

    with tf.Graph().as_default():
        build_model(hyperparameters)

        with tf.device(TF_DEVICE), tf.Session() as sess:
            tf.train.Saver().restore(sess, checkpoint_filename+".ckpt")
            for i in range(0,n_samples,BATCH_SIZE):
                pred[i:i+BATCH_SIZE], conf[i:i+BATCH_SIZE], offset[i:i+BATCH_SIZE], coldensity[i:i+BATCH_SIZE] = \
                    sess.run([t('prediction'), t('output_classifier'), t('y_nn_offset'), t('y_nn_coldensity')],
                             feed_dict={t('x'):                 flux[i:i+BATCH_SIZE,:],
                                        t('keep_prob'):         1.0})

    print "Localize Model processed %d samples in chunks of %d in %0.1f seconds" % \
          (n_samples, BATCH_SIZE, timeit.default_timer() - timer)

    # coldensity_rescaled = coldensity * COL_DENSITY_STD + COL_DENSITY_MEAN
    return pred, conf, offset, coldensity


# # Implementation, internal function used for parallel map processing, call predictions_to_central_wavelength
# def _predictions_to_central_wavelength((prediction_confidences, offsets, samples_per_sightline, min_width, max_width)):
#     MIN_THRESHOLD = 0.05
#
#     results = []       # Returns a list of tuples: (peaks_centered, peaks_uncentered, smoothed_conf)
#     num_sightlines = int(prediction_confidences.shape[0] / samples_per_sightline)
#     gc.disable()
#     for i in range(0, num_sightlines):
#         sample_range = np.shape(prediction_confidences)[0] / num_sightlines
#         ix_samples = i * sample_range
#         sample = prediction_confidences[ix_samples:ix_samples + sample_range]
#         sample_offset = offsets[ix_samples:ix_samples + sample_range]
#
#         widths = np.arange(min_width, max_width)
#         peaks_all = find_peaks_cwt(sample, widths, max_distances=widths / 4,
#                                           noise_perc=1, gap_thresh=2, min_length=25, min_snr=1)
#
#         # Translate relative offsets to histogram
#         offset_to_ix = np.arange(len(sample_offset)) - sample_offset
#         offset_to_ix[offset_to_ix < 0] = 0
#         offset_to_ix[offset_to_ix >= len(sample_offset)] = len(sample_offset)
#         offset_hist, ignore_offset_range = np.histogram(offset_to_ix, bins=np.arange(0,len(sample_offset)+1))
#
#         # Old deprecated peaks detection - to be removed once dependencies are eliminated
#         # Center peaks using half-width approach
#         smoothed_sample = signal.medfilt(sample, 75)
#         peaks_uncentered = []
#         peaks_centered = []
#         ixs_left = []
#         ixs_right = []
#         for peak in peaks_all:
#             logical_array_half_peak = np.pad(smoothed_sample >= smoothed_sample[peak] / 2, (1, 1), 'constant')
#             ix_left = peak - np.nonzero(np.logical_not(np.flipud(logical_array_half_peak[0:peak + 1])))[0][0]
#             ix_right = peak + np.nonzero(np.logical_not(logical_array_half_peak[peak + 1:]))[0][0] - 1
#             assert ix_right < np.shape(sample)[0] and ix_right >= 0, "ix_right [%d] out of range: %d" % \
#                                                                      (ix_right, np.shape(sample)[0])
#             assert ix_left >= 0, "ix_left [%d] out of range" % (ix_left)
#             peak_centered = int(ix_left + (ix_right - ix_left) / 2)
#             # Save peak only if it exceeds the minimum threshold
#             if smoothed_sample[peak_centered] > MIN_THRESHOLD:
#                 peaks_uncentered.append(peak)
#                 peaks_centered.append(peak_centered)
#                 ixs_left.append(ix_left)
#                 ixs_right.append(ix_right)
#
#         # offset localization histogram, convolved sum plotline, and peaks
#         offset_hist = offset_hist / 80.0
#         po = np.pad(offset_hist, 2, 'constant', constant_values=np.mean(offset_hist))
#         offset_conv_sum = (po[:-4] + po[1:-3] + po[2:-2] + po[3:-1] + po[4:])
#         smooth_conv_sum = signal.medfilt(offset_conv_sum, 9)
#         peaks_interp = scipy.signal.find_peaks_cwt(smooth_conv_sum, np.arange(1, 100))
#
#         # peaks_all = peakutils.indexes(offset_conv_sum, thres=0.1, min_dist=60)
#         # peaks_interp = scipy.signal.find_peaks_cwt(smooth_conv_sum, np.arange(1,100))
#
#         # # Fine tune the location by fitting a gaussian with peakutils.interpolate
#         # # Interpolate will sometimes fail: http://stackoverflow.com/questions/9172574/scipy-curve-fit-runtime-error-stopping-iteration
#         # # We capture that and use the original values when that occurs
#         # try:
#         #     print "    ", peaks_all
#         #     peaks_interp = peakutils.interpolate(np.arange(0, len(offset_conv_sum)),
#         #                                      offset_conv_sum, ind=peaks_all, width=10).astype(int)
#         # except RuntimeError as e:
#         #     print('peakutils.interpolate failed to find a solution, continuing with the original values', e)
#         #     peaks_interp = peaks_all
#         # # Validate interpolate results, the algorithm is buggy and can produce invalid results (negative numbers!)
#         # if np.any(np.logical_or(np.array(peaks_interp) <= 40, np.array(peaks_interp) >= len(offset_conv_sum) - 40)):
#         #     print('peakutils.interpolate produced invalid results, using raw indexes instead')
#         #     peaks_offset = peaks_all
#         # else:
#         #     peaks_offset = peaks_interp
#
#         # REST_RANGE was expanded by 40 pixels on each side to accommodate dropping the lowest and highest ranges
#         # so that peak detection and interpolate have a valid range of data to work with at the low & high end
#         peaks_filtered = [peak for peak in peaks_interp if offset_conv_sum[peak] > 0.2 and
#                                                         40 <= peak <= len(offset_conv_sum) - 40]
#
#         # peaks_offset = np.array(peaks_interp)
#         peaks_offset = np.array(peaks_filtered, dtype=np.int32)
#         assert all(peaks_offset <= offset_hist.shape[0])     # Make sure no peaks run off the right side
#         results.append((peaks_centered, peaks_uncentered, smoothed_sample, ixs_left, ixs_right,
#                         offset_hist, offset_conv_sum, peaks_offset))
#         np.set_printoptions(threshold=np.nan)
#         # print peaks_offset, "peaks_offset (final)\n  peaks_interp: ", peaks_interp, "\n  peaks_filtered: ", peaks_filtered, "\n  peaks_all:", peaks_all, offset_conv_sum[peaks_filtered]#, '\n', offset_conv_sum
#         # np.save('../tmp/tmp.npy', offset_conv_sum)
#     gc.enable()
#     return results




# Returns the index location of the peaks along prediction_confidences line
# RETURN VALUE: An array of tuples for each sightline processed, in the form:
#    [
#     (peaks_centered, peaks_uncentered, smoothed_sample, ixs_left, ixs_right),
#     (peaks_centered, peaks_uncentered, smoothed_sample, ixs_left, ixs_right),
#      ...
#    ]
# def predictions_to_central_wavelength(prediction_confidences, offsets, num_sightlines, min_width=100, max_width=360):
#     cores = multiprocessing.cpu_count() - 1
#     p = multiprocessing.Pool(cores)
#     n_samples = prediction_confidences.shape[0]
#     samples_per_sightline = prediction_confidences.shape[0] / num_sightlines
#     sightline_split = int(math.ceil(float(num_sightlines) / cores))
#     split_range = range(sightline_split*samples_per_sightline, n_samples, sightline_split*samples_per_sightline)
#     list_prediction_confidences = np.split(prediction_confidences, split_range)
#     list_offsets = np.split(offsets, split_range)
#     multiprocessing_params = np.ones((len(list_prediction_confidences),))
#     list_results = map(_predictions_to_central_wavelength, #todo p.map
#                          zip(list_prediction_confidences,
#                              list_offsets,
#                              multiprocessing_params * samples_per_sightline,  # copy params to every map call
#                              multiprocessing_params * min_width,
#                              multiprocessing_params * max_width))
#     p.close()
#     p.join()
#
#     results = [item for sublist in list_results for item in sublist]        # Flatten list of lists
#     return results


def train_ann(hyperparameters, train_dataset, test_dataset, save_filename=None, load_filename=None, tblogs = "../tmp/tblogs", TF_DEVICE=''):
    training_iters = hyperparameters['training_iters']
    batch_size = hyperparameters['batch_size']
    dropout_keep_prob = hyperparameters['dropout_keep_prob']

    # Load dataset
    # data_train = load_dataset(train_dataset_filename)
    # data_test = load_dataset(test_dataset_filename)
    # positive_sample_indexes = np.nonzero(data_train['labels_classifier'] == 1)[0]
    # batch_iterator = BatchIterator(data_train['fluxes'].shape[0])
    # batch_iterator_pos_ABC = BatchIterator(positive_sample_indexes.shape[0])


    # Predefine variables that need to be returned from local scope
    best_accuracy = 0.0
    best_offset_rmse = 999999999
    best_density_rmse = 999999999
    test_accuracy = None
    loss_value = None

    with tf.Graph().as_default():
        # Build model
        # (train_step_AB, train_step_ABC, accuracy, loss_classifier, loss_offset_regression, loss_coldensity_regression, x, label_classifier, label_offset,
        #  label_coldensity, keep_prob, prediction, output_classifier, y_nn_offset, rmse_offset, y_nn_coldensity,
        #  rmse_coldensity) = build_model(hyperparameters)
        train_step_ABC, tb_summaries = build_model(hyperparameters)

        with tf.device(TF_DEVICE), tf.Session() as sess:
            # Restore or initialize model
            if load_filename is not None:
                tf.train.Saver().restore(sess, load_filename+".ckpt")
                print("Model loaded from checkpoint: %s"%load_filename)
            else:
                print "Initializing variables"
                sess.run(tf.initialize_all_variables())

            summary_writer = tf.train.SummaryWriter(tblogs, sess.graph)

            for i in range(training_iters):
                # if False: #i % 8 == 0:       # Alternate between training 2 loss functions vs. 3
                #     batch_ix = positive_sample_indexes[batch_iterator_pos_ABC.next_batch(batch_size)]
                #     sess.run(train_step_ABC, feed_dict={t('x'):                data_train['fluxes'][batch_ix],
                #                                         t('label_classifier'): data_train['labels_classifier'][batch_ix],
                #                                         t('label_offset'):     data_train['labels_offset'][batch_ix],
                #                                         t('label_coldensity'): data_train['col_density'][batch_ix],
                #                                         t('keep_prob'):        dropout_keep_prob})
                #
                # else:
                # batch_ix = batch_iterator.next_batch(batch_size)
                batch_fluxes, batch_labels_classifier, batch_labels_offset, batch_col_density = train_dataset.next_batch(batch_size)
                sess.run(train_step_ABC, feed_dict={t('x'):                batch_fluxes,
                                                    t('label_classifier'): batch_labels_classifier,
                                                    t('label_offset'):     batch_labels_offset,
                                                    t('label_coldensity'): batch_col_density,
                                                    t('keep_prob'):        dropout_keep_prob})

                if i % 200 == 0:   # i%2 = 1 on 200ths iteration, that's important so we have the full batch pos/neg
                    train_accuracy, loss_value, result_rmse_offset, result_loss_offset_regression, \
                    result_rmse_coldensity, result_loss_coldensity_regression \
                        = train_ann_test_batch(sess, np.random.permutation(train_dataset.fluxes.shape[0])[0:10000],
                                               train_dataset.data, summary_writer=summary_writer)
                        # = train_ann_test_batch(sess, batch_ix, data_train)  # Note this batch_ix must come from train_step_ABC
                    print "step %06d, classify-offset-density acc/loss - RMSE/loss      %0.3f/%0.3f - %0.3f/%0.3f - %0.3f/%0.3f" \
                          % (i, train_accuracy, float(np.mean(loss_value)), result_rmse_offset,
                             result_loss_offset_regression, result_rmse_coldensity,
                             result_loss_coldensity_regression)
                if i % 5000 == 0 or i == training_iters - 1:
                    test_accuracy, _, result_rmse_offset, _, result_rmse_coldensity, _ = train_ann_test_batch(
                        sess, np.arange(test_dataset.fluxes.shape[0]), test_dataset.data)
                    best_accuracy = test_accuracy if test_accuracy > best_accuracy else best_accuracy
                    best_offset_rmse = result_rmse_offset if result_rmse_offset < best_offset_rmse else best_offset_rmse
                    best_density_rmse = result_rmse_coldensity if result_rmse_coldensity < best_density_rmse else best_density_rmse
                    print "             test accuracy/offset RMSE/density RMSE:     %0.3f / %0.3f / %0.3f" % \
                          (test_accuracy, result_rmse_offset, result_rmse_coldensity)
                    save_checkpoint(sess, save_filename + "_" + str(i))

           # Save checkpoint
            save_checkpoint(sess, save_filename)

            return best_accuracy, test_accuracy, np.mean(loss_value), best_offset_rmse, result_rmse_offset, \
                   best_density_rmse, result_rmse_coldensity


def save_checkpoint(sess, save_filename):
    if save_filename is not None:
        tf.train.Saver().save(sess, save_filename + ".ckpt")
        with open(checkpoint_filename + "_hyperparams.json", 'w') as fp:
            json.dump(hyperparameters, fp)
        print("Model saved in file: %s" % save_filename + ".ckpt")


# Get a tensor by name, convenience method
def t(tensor_name):
    tensor_name = tensor_name+":0" if not tensor_regex.match(tensor_name) else tensor_name
    return tf.get_default_graph().get_tensor_by_name(tensor_name)


# Called from train_ann to perform a test of the train or test data, needs to separate pos/neg to get accurate #'s
def train_ann_test_batch(sess, ixs, data, summary_writer=None):    #inputs: label_classifier, label_offset, label_coldensity
    MAX_BATCH_SIZE = 40000.0
    # Create a global counter for the summary step outputs
    # global summary_step
    # try: summary_step
    # except NameError: summary_step = 0

    classifier_accuracy = 0.0
    classifier_loss_value = 0.0
    result_rmse_offset = 0.0
    result_loss_offset_regression = 0.0
    result_rmse_coldensity = 0.0
    result_loss_coldensity_regression = 0.0

    # split x and labels into pos/neg
    pos_ixs = ixs[data['labels_classifier'][ixs]==1]
    neg_ixs = ixs[data['labels_classifier'][ixs]==0]
    pos_ixs_split = np.array_split(pos_ixs, math.ceil(len(pos_ixs)/MAX_BATCH_SIZE)) if len(pos_ixs) > 0 else []
    neg_ixs_split = np.array_split(neg_ixs, math.ceil(len(neg_ixs)/MAX_BATCH_SIZE)) if len(neg_ixs) > 0 else []

    sum_samples = float(len(ixs))       # forces cast of percent len(pos_ix)/sum_samples to a float value
    sum_pos_only = float(len(pos_ixs))

    # Process pos and neg samples separately
    # pos
    for pos_ix in pos_ixs_split:
        tensors = [t('accuracy'), t('loss_classifier'), t('rmse_offset'), t('loss_offset_regression'),
                        t('rmse_coldensity'), t('loss_coldensity_regression'), t('global_step')]
        if summary_writer is not None:
            tensors.extend(tf.get_collection('SUMMARY_A'))
            tensors.extend(tf.get_collection('SUMMARY_B'))
            tensors.extend(tf.get_collection('SUMMARY_C'))
            tensors.extend(tf.get_collection('SUMMARY_shared'))

        run = sess.run(tensors, feed_dict={t('x'):                data['fluxes'][pos_ix],
                                           t('label_classifier'): data['labels_classifier'][pos_ix],
                                           t('label_offset'):     data['labels_offset'][pos_ix],
                                           t('label_coldensity'): data['col_density'][pos_ix],
                                           t('keep_prob'):        1.0})
        weight_wrt_total_samples = len(pos_ix)/sum_samples
        weight_wrt_pos_samples = len(pos_ix)/sum_pos_only
        # print "DEBUG> Processing %d positive samples" % len(pos_ix), weight_wrt_total_samples, sum_samples, weight_wrt_pos_samples, sum_pos_only
        classifier_accuracy += run[0] * weight_wrt_total_samples
        classifier_loss_value += np.sum(run[1]) * weight_wrt_total_samples
        result_rmse_offset += run[2] * weight_wrt_total_samples
        result_loss_offset_regression += run[3] * weight_wrt_total_samples
        result_rmse_coldensity += run[4] * weight_wrt_pos_samples
        result_loss_coldensity_regression += run[5] * weight_wrt_pos_samples

        for i in range(7, len(tensors)):
            summary_writer.add_summary(run[i], run[6])

    # neg
    for neg_ix in neg_ixs_split:
        tensors = [t('accuracy'), t('loss_classifier'), t('rmse_offset'), t('loss_offset_regression'), t('global_step')]
        if summary_writer is not None:
            tensors.extend(tf.get_collection('SUMMARY_A'))
            tensors.extend(tf.get_collection('SUMMARY_B'))
            tensors.extend(tf.get_collection('SUMMARY_shared'))

        run = sess.run(tensors, feed_dict={t('x'):                data['fluxes'][neg_ix],
                                           t('label_classifier'): data['labels_classifier'][neg_ix],
                                           t('label_offset'):     data['labels_offset'][neg_ix],
                                           t('keep_prob'):        1.0})
        weight_wrt_total_samples = len(neg_ix)/sum_samples
        # print "DEBUG> Processing %d negative samples" % len(neg_ix), weight_wrt_total_samples, sum_samples
        classifier_accuracy += run[0] * weight_wrt_total_samples
        classifier_loss_value += np.sum(run[1]) * weight_wrt_total_samples
        result_rmse_offset += run[2] * weight_wrt_total_samples
        result_loss_offset_regression += run[3] * weight_wrt_total_samples

        for i in range(5,len(tensors)):
            summary_writer.add_summary(run[i], run[4])

    return classifier_accuracy, classifier_loss_value, \
           result_rmse_offset, result_loss_offset_regression, \
           result_rmse_coldensity, result_loss_coldensity_regression



if __name__ == '__main__':
    #
    # Execute batch mode
    #
    DEFAULT_TRAINING_ITERS = 30000
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--hyperparamsearch', help='Run hyperparam search', required=False, action='store_true', default=False)
    parser.add_argument('-o', '--output_file', help='Hyperparam csv result file location', required=False, default='../tmp/batch_results.csv')
    parser.add_argument('-i', '--iterations', help='Number of training iterations', required=False, default=DEFAULT_TRAINING_ITERS)
    parser.add_argument('-l', '--loadmodel', help='Specify a model name to load', required=False, default=None)
    parser.add_argument('-c', '--checkpoint_file', help='Name of the checkpoint file to save (without file extension)', required=False, default="../models/localize_model")
    parser.add_argument('-r', '--train_dataset_filename', help='File name of the training dataset without extension', required=False, default="../data/gensample/train_*.pickle")
    parser.add_argument('-e', '--test_dataset_filename', help='File name of the testing dataset without extension', required=False, default="../data/gensample/test_96451_5000.pickle")
    args = vars(parser.parse_args())

    RUN_SINGLE_ITERATION = not args['hyperparamsearch']
    checkpoint_filename = args['checkpoint_file'] if RUN_SINGLE_ITERATION else None
    batch_results_file = args['output_file']
    tf.logging.set_verbosity(tf.logging.DEBUG)

    train_dataset = Dataset(args['train_dataset_filename'])
    test_dataset = Dataset(args['test_dataset_filename'])

    exception_counter = 0
    iteration_num = 0

    parameter_names = ["learning_rate", "training_iters", "batch_size", "l2_regularization_penalty", "dropout_keep_prob",
                       "fc1_n_neurons", "fc2_1_n_neurons", "fc2_2_n_neurons", "fc2_3_n_neurons",
                       "conv1_kernel", "conv2_kernel", "conv1_filters", "conv2_filters", "conv1_stride", "conv2_stride",
                       "pool1_kernel", "pool2_kernel", "pool1_stride", "pool2_stride",
                       "pool1_method", "pool2_method"]
    parameters = [
        # First column: Keeps the best best parameter based on accuracy score
        # Other columns contain the parameter options to try

        # learning_rate
        [0.0005,         0.0005, 0.0007, 0.0010, 0.0030, 0.0050, 0.0070],
        # training_iters
        [int(args['iterations'])],
        # batch_size
        [700,           400, 500, 600, 700, 850, 1000],
        # l2_regularization_penalty
        [0.005,         0.08, 0.01, 0.008, 0.005, 0.003],
        # dropout_keep_prob
        [0.98,          0.5, 0.75, 0.9, 0.95, 0.98, 1],
        # fc1_n_neurons
        [350,           50, 75, 100, 150, 200, 350, 500],
        # fc2_1_n_neurons
        [200,           50, 75, 100, 150, 200, 350, 500],
        # fc2_2_n_neurons
        [350,           50, 75, 100, 150, 200, 350, 500],
        # fc2_3_n_neurons
        [150,           50, 75, 100, 150, 200, 350, 500],
        # conv1_kernel
        [32,            8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32],
        # conv2_kernel
        [16,            14, 16, 20, 24, 28, 32, 34],
        # conv1_filters
        [100,           64, 80, 90, 100, 110, 120, 140],
        # conv2_filters
        [96,            80, 96, 128, 192, 256],
        # conv1_stride
        [3,             2, 3, 4, 5, 6, 8],
        # conv2_stride
        [1,             1, 2, 3, 4, 5, 6],
        # pool1_kernel
        [7,             3, 4, 5, 6, 7, 8],
        # pool2_kernel
        [6,             4, 5, 6, 7, 8, 9, 10],
        # pool1_stride
        [4,             1, 2, 4, 5, 6],
        # pool2_stride
        [4,             1, 2, 3, 4, 5, 6, 7, 8],
        # pool1_method
        [1,             1, 2],
        #pool2_method
        [1,             1, 2]
    ]

    # Write out CSV header
    os.remove(batch_results_file) if os.path.exists(batch_results_file) else None
    with open(batch_results_file, "a") as csvoutput:
        csvoutput.write("iteration_num,normalized_score,best_accuracy,last_accuracy,last_objective,best_offset_rmse,last_offset_rmse,best_coldensity_rmse,last_coldensity_rmse," + ",".join(parameter_names) + "\n")

    while (RUN_SINGLE_ITERATION and iteration_num < 1) or not RUN_SINGLE_ITERATION:
        iteration_best_result = 9999999

        i = random.randint(0,len(parameters)-1)           # Choose a random parameter to change
        hyperparameters = {}    # Dictionary that stores all hyperparameters used in this iteration

        for j in (range(1,len(parameters[i])) if not RUN_SINGLE_ITERATION else range(0,1)):
            iteration_num += 1

            for k in range(0,len(parameter_names)):
                hyperparameters[parameter_names[k]] = parameters[k][j] if i == k else parameters[k][0]

            try:
                # Print parameters during training
                print("----------------------------------------------------------------------------------------------")
                print("PARAMETERS (varying %s): " % parameter_names[i])
                for k in range(0,len(parameters)):
                    sys.stdout.write("{:<30}{:<15}\n".format( parameter_names[k], parameters[k][j] if k==i else parameters[k][0] ))

                # ANN Training
                (best_accuracy, last_accuracy, last_objective, best_offset_rmse, last_offset_rmse, best_coldensity_rmse,
                 last_coldensity_rmse) = train_ann(hyperparameters, train_dataset, test_dataset,
                                                   save_filename=checkpoint_filename, load_filename=args['loadmodel'])

                # These mean & std dev are used to normalize the score from all 3 loss functions for hyperparam optimize
                mean_best_accuracy = 0.02
                std_best_accuracy = 0.0535
                mean_best_offset_rmse = 4.0
                std_best_offset_rmse = 1.56
                mean_best_coldensity_rmse = 0.33
                std_best_coldensity_rmse = 0.1
                normalized_score = (1-best_accuracy-mean_best_accuracy)/std_best_accuracy + \
                                   (best_offset_rmse-mean_best_offset_rmse)/std_best_offset_rmse + \
                                   (best_coldensity_rmse-mean_best_coldensity_rmse)/std_best_coldensity_rmse

                # Save results and parameters to CSV
                with open(batch_results_file, "a") as csvoutput:
                    csvoutput.write("%d,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f\n" % \
                                    (iteration_num, normalized_score, best_accuracy, last_accuracy,
                                     float(last_objective), best_offset_rmse, last_offset_rmse,
                                     best_coldensity_rmse, last_coldensity_rmse,
                                     hyperparameters['learning_rate'], hyperparameters['training_iters'],
                                     hyperparameters['batch_size'], hyperparameters['l2_regularization_penalty'],
                                     hyperparameters['dropout_keep_prob'], hyperparameters['fc1_n_neurons'],
                                     hyperparameters['fc2_1_n_neurons'], hyperparameters['fc2_2_n_neurons'],
                                     hyperparameters['fc2_3_n_neurons'],
                                     hyperparameters['conv1_kernel'], hyperparameters['conv2_kernel'],
                                     hyperparameters['conv1_filters'], hyperparameters['conv2_filters'],
                                     hyperparameters['conv1_stride'], hyperparameters['conv2_stride'],
                                     hyperparameters['pool1_kernel'], hyperparameters['pool2_kernel'],
                                     hyperparameters['pool1_stride'], hyperparameters['pool2_stride'],
                                     hyperparameters['pool1_method'], hyperparameters['pool2_method']) )

                # Keep a running tab of the best parameters based on overall accuracy
                if normalized_score <= iteration_best_result:
                    iteration_best_result = normalized_score
                    parameters[i][0] = parameters[i][j]
                    print("Best result for parameter [%s] with iteration_best_result [%0.2f] now set to [%f]" % (parameter_names[i], iteration_best_result, parameters[i][0]))

                exception_counter = 0   # reset exception counter on success

            # Log and ignore exceptions
            except Exception as e:
                exception_counter += 1
                with open(batch_results_file, "a") as csvoutput:
                    csvoutput.write("%d,error,error,error,error,error,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f\n" % \
                                    (iteration_num,
                                     hyperparameters['learning_rate'], hyperparameters['training_iters'],
                                     hyperparameters['batch_size'], hyperparameters['l2_regularization_penalty'],
                                     hyperparameters['dropout_keep_prob'], hyperparameters['fc1_n_neurons'],
                                     hyperparameters['conv1_kernel'], hyperparameters['conv2_kernel'],
                                     hyperparameters['conv1_filters'], hyperparameters['conv2_filters'],
                                     hyperparameters['conv1_stride'], hyperparameters['conv2_stride'],
                                     hyperparameters['pool1_kernel'], hyperparameters['pool2_kernel'],
                                     hyperparameters['pool1_stride'], hyperparameters['pool2_stride'],
                                     hyperparameters['pool1_method'], hyperparameters['pool2_method']) )

                traceback.print_exc()
                if exception_counter > 20:
                    exit()                  # circuit breaker



