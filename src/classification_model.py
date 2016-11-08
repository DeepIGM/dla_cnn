#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import random, os, sys, traceback, math, json
from DataSet import DataSet

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


def build_model(hyperparameters):
    learning_rate = hyperparameters['learning_rate']
    l2_regularization_penalty = hyperparameters['l2_regularization_penalty']
    fc1_n_neurons = hyperparameters['fc1_n_neurons']
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

    INPUT_SIZE = 1696

    x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
    y_ = tf.placeholder(tf.float32, shape=[None])

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

    # Densely Connected Layer
    inputsize_fc3 = int(math.ceil(math.ceil(
        math.ceil(math.ceil(INPUT_SIZE / conv1_stride) / pool1_stride) / conv2_stride) / pool2_stride)) * conv2_filters

    W_fc3 = weight_variable([inputsize_fc3, fc1_n_neurons])
    b_fc3 = bias_variable([fc1_n_neurons])

    h_pool2_flat = tf.reshape(h_pool2, [-1, inputsize_fc3])
    h_fc3 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc3) + b_fc3)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

    # Readout Layer
    W_fc4 = weight_variable([fc1_n_neurons, 1])
    b_fc4 = bias_variable([1])

    y_fc4 = tf.add(tf.matmul(h_fc3_drop, W_fc4), b_fc4)
    y_nn = tf.reshape(y_fc4, [-1])

    # Train and Evaluate the model
    cost = tf.nn.sigmoid_cross_entropy_with_logits(y_nn, y_) + \
           l2_regularization_penalty * tf.nn.l2_loss(W_conv1) + \
           l2_regularization_penalty * tf.nn.l2_loss(W_conv2) + \
           l2_regularization_penalty * tf.nn.l2_loss(W_fc3) + \
           l2_regularization_penalty * tf.nn.l2_loss(W_fc4)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    output = tf.sigmoid(y_nn)
    prediction = tf.round(output)
    correct_prediction = tf.equal(prediction, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return train_step, accuracy, cost, y_, x, keep_prob, prediction, output

def predictions_ann_multiprocess(param_tuple):
    return predictions_ann(param_tuple[0], param_tuple[1], param_tuple[2], param_tuple[3])


def predictions_ann(hyperparameters, flux, labels, checkpoint_filename, TF_DEVICE=''):
    with tf.Graph().as_default(), tf.device(TF_DEVICE), tf.Session() as sess:
        train_step, accuracy, cost, y_, x, keep_prob, prediction, output = build_model(hyperparameters)

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_filename + ".ckpt")
        pred, conf = sess.run([prediction, output], feed_dict={x: flux, y_: labels, keep_prob: 1.0})

    return pred, conf


def load_data_sets(nan=0.0, dtype=np.float32):
    train_flux = np.load("../data/classification_train.npy")
    test_dla_flux = np.load("../data/classification_test_dla.npy")
    test_non_dla_flux = np.load("../data/classification_test_non_dla.npy")

    # set nan parameter
    train_flux[np.isnan(train_flux)] = nan
    test_dla_flux[np.isnan(test_dla_flux)] = nan
    test_non_dla_flux[np.isnan(test_non_dla_flux)] = nan

    return DataSet(train_flux), DataSet(test_dla_flux), DataSet(test_non_dla_flux)


def train_ann(hyperparameters, save_filename=None, load_filename=None):
    training_iters = hyperparameters['training_iters']
    batch_size = hyperparameters['batch_size']
    dropout_keep_prob = hyperparameters['dropout_keep_prob']
    nan_parameter = hyperparameters['nan_parameter']

    train, test_dla, test_non_dla = load_data_sets(nan=nan_parameter)

    # Predefine variables that need to be returned from local scope
    best_accuracy = 0.0
    mean_accuracy = None
    accuracy_DLAs = None
    accuracy_non_DLAs = None
    loss_value = None

    with tf.Graph().as_default():
        # Build model
        train_step, accuracy, cost, y_, x, keep_prob, prediction, output = build_model(hyperparameters)

        with tf.Session() as sess:
            # with tf.device():
            # Restore or initialize model
            if load_filename is not None:
                tf.train.Saver().restore(sess, load_filename + ".ckpt")
                print("Model loaded from checkpoint: %s" % load_filename)
            else:
                sess.run(tf.initialize_all_variables())

            for i in range(training_iters):
                batch = train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: dropout_keep_prob})
                if i % 200 == 0:
                    train_accuracy, loss_value = sess.run([accuracy, cost],
                                                          feed_dict={x: train.fluxes, y_: train.labels, keep_prob: 1.0})
                    print(
                        "step %06d, training accuracy/loss      %0.3f / %0.3f" % (
                            i, train_accuracy, np.mean(loss_value)))
                if i % 1000 == 0 or i == training_iters - 1:
                    accuracy_DLAs = sess.run(accuracy,
                                             feed_dict={x: test_dla.fluxes, y_: test_dla.labels, keep_prob: 1.0})
                    accuracy_non_DLAs = sess.run(accuracy, feed_dict={x: test_non_dla.fluxes, y_: test_non_dla.labels,
                                                                      keep_prob: 1.0})
                    mean_accuracy = np.mean([accuracy_DLAs, accuracy_non_DLAs])
                    best_accuracy = mean_accuracy if (mean_accuracy > best_accuracy) else best_accuracy
                    print("             test accuracy for DLAs:     %0.3f" % accuracy_DLAs)
                    print("             test accuracy for non-DLAs: %0.3f" % accuracy_non_DLAs)
                    print("             avg accuracy (%06d):      %0.3f" % (i, mean_accuracy))

            # Save checkpoint
            if save_filename is not None:
                saver = tf.train.Saver()
                save_path = saver.save(sess, save_filename + ".ckpt")
                with open(save_filename + "_hyperparams.json", 'w') as fp:
                    json.dump(hyperparameters, fp)
                print("Model saved in file: %s" % save_path)

    return best_accuracy, mean_accuracy, accuracy_DLAs, accuracy_non_DLAs, np.mean(loss_value)


if __name__ == '__main__':
    #
    # Execute batch mode
    #
    RUN_SINGLE_ITERATION = True
    iteration_num = 0
    checkpoint_filename = "../models/classification_model"

    parameter_names = ["learning_rate", "training_iters", "batch_size", "l2_regularization_penalty",
                       "dropout_keep_prob",
                       "fc1_n_neurons", "conv1_kernel", "conv2_kernel", "conv1_filters", "conv2_filters",
                       "conv1_stride", "conv2_stride", "pool1_kernel", "pool2_kernel", "pool1_stride", "pool2_stride",
                       "nan_parameter", "pool1_method", "pool2_method"]
    parameters = [
        # First column: Keeps the best best parameter based on accuracy score
        # Other columns contain the parameter options to try

        # learning_rate
        [5e-3, 0.0005, 0.001, 0.003, 0.007, 0.01],
        # training_iters
        [15000],
        # batch_size
        [50, 50, 75, 100, 150],
        # l2_regularization_penalty
        [0.005, 0.08, 0.01, 0.008, 0.005, 0.003],
        # dropout_keep_prob
        [0.5, 0.5, 0.75, 0.9, 0.95, 0.98, 1],
        # fc1_n_neurons
        [50, 50, 75, 100, 150, 200, 500],
        # conv1_kernel
        [18, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32],
        # conv2_kernel
        [16, 4, 8, 16, 20, 24, 27, 28, 29, 30, 32, 34],
        # conv1_filters
        [80, 8, 16, 24, 32, 48, 64, 80],
        # conv2_filters
        [96, 32, 48, 64, 80, 96, 128],
        # conv1_stride
        [4, 2, 3, 4, 5, 6, 8],
        # conv2_stride
        [1, 1, 2, 3, 4, 5, 6],
        # pool1_kernel
        [6, 3, 4, 5, 6, 7, 8],
        # pool2_kernel
        [7, 4, 5, 6, 7, 8, 9, 10],
        # pool1_stride
        [2, 1, 2, 4, 5, 6],
        # pool2_stride
        [2, 1, 2, 3, 4, 5, 6, 7, 8],
        # nan_parameter
        [0.0, -5.0, 0.0, 5.0, 10.0],
        # pool1_method
        [1, 1, 2],
        # pool2_method
        [1, 1, 2]
    ]

    # Write out CSV header
    batch_results_file = '../tmp/batch_results.csv'
    os.remove(batch_results_file) if os.path.exists(batch_results_file) else None
    with open(batch_results_file, "a") as csvoutput:
        csvoutput.write(
            "iteration_num,best_accuracy,last_accuracy,accuracy_DLAs,accuracy_non_DLAs,last_objective," + ",".join(
                parameter_names) + "\n")

    while 1:
        if RUN_SINGLE_ITERATION and iteration_num >= 1:
            sys.exit()
        iteration_best_accuracy = 0.0

        i = random.randint(0, len(parameters) - 1)  # Choose a random parameter to change
        hyperparameters = {}  # Dictionary that stores all hyperparameters used in this iteration

        # Gather all hyperparameters for this particular iteration into a dictionary called hyperaparameters
        for j in (range(1, len(parameters[i])) if not RUN_SINGLE_ITERATION else range(0, 1)):
            iteration_num += 1

            for k in range(0, len(parameter_names)):
                hyperparameters[parameter_names[k]] = parameters[k][j] if i == k else parameters[k][0]

            try:
                # Print parameters during training
                print("----------------------------------------------------------------------------------------------")
                print("PARAMETERS (varying %s): " % parameter_names[i])
                for k in range(0, len(parameters)):
                    sys.stdout.write("{:<30}{:<15}\n".format(parameter_names[k], hyperparameters[
                        parameter_names[k]]))  # parameters[k][j] if k==i else parameters[k][0] ))

                # ANN Training
                (best_accuracy, last_accuracy, accuracy_DLAs, accuracy_non_DLAs, last_objective) \
                    = train_ann(hyperparameters, checkpoint_filename, None)

                # Save results and parameters to CSV
                with open("batch_results.csv", "a") as csvoutput:
                    csvoutput.write(
                        "%d,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f\n" % \
                        (iteration_num, best_accuracy, last_accuracy,
                         accuracy_DLAs, accuracy_non_DLAs, last_objective,
                         hyperparameters['learning_rate'], hyperparameters['training_iters'],
                         hyperparameters['batch_size'], hyperparameters['l2_regularization_penalty'],
                         hyperparameters['dropout_keep_prob'], hyperparameters['fc1_n_neurons'],
                         hyperparameters['conv1_kernel'], hyperparameters['conv2_kernel'],
                         hyperparameters['conv1_filters'], hyperparameters['conv2_filters'],
                         hyperparameters['conv1_stride'], hyperparameters['conv2_stride'],
                         hyperparameters['pool1_kernel'], hyperparameters['pool2_kernel'],
                         hyperparameters['pool1_stride'], hyperparameters['pool2_stride'],
                         hyperparameters['nan_parameter'], hyperparameters['pool1_method'],
                         hyperparameters['pool2_method']))

                # Keep a running tab of the best parameters based on overall accuracy
                if best_accuracy >= iteration_best_accuracy:
                    iteration_best_accuracy = best_accuracy
                    parameters[i][0] = parameters[i][j]
                    print("Best accuracy for parameter [%s] with accuracy [%0.2f] now set to [%f]" % (
                        parameter_names[i], iteration_best_accuracy, parameters[i][0]))

                if not RUN_SINGLE_ITERATION:
                    # Save predictions
                    p_train, p_test_dla, p_test_non_dla = load_data_sets(nan=hyperparameters['nan_parameter'])
                    predictions_train, confidence_train = predictions_ann(hyperparameters, p_train.fluxes,
                                                                          p_train.labels, checkpoint_filename)
                    predictions_test_DLAs, confidence_test_DLAs = predictions_ann(hyperparameters, p_test_dla.fluxes,
                                                                                  p_test_dla.labels,
                                                                                  checkpoint_filename)
                    predictions_test_non_DLAs, confidence_test_non_DLAs = predictions_ann(hyperparameters,
                                                                                          p_test_non_dla.fluxes,
                                                                                          p_test_non_dla.labels,
                                                                                          checkpoint_filename)

                    p, c = predictions_ann(hyperparameters, np.reshape(p_test_dla.fluxes[27, :], (1, 1024)),
                                           np.reshape(p_test_dla.labels[27], (1,)), checkpoint_filename)
                    print(p, c)

                    with open('predictions_train.npy', 'w') as file_pred_train:
                        np.save(file_pred_train, predictions_train)
                    with open('predictions_test_DLAs.npy', 'w') as file_pred_dlas:
                        np.save(file_pred_dlas, predictions_test_DLAs)
                    with open('predictions_test_nonDLAs.npy', 'w') as file_pred_nondlas:
                        np.save(file_pred_nondlas, predictions_test_non_DLAs)
                    with open('confidence_train.npy', 'w') as file_conf_train:
                        np.save(file_conf_train, confidence_train)
                    with open('confidence_test_DLAs.npy', 'w') as file_conf_dlas:
                        np.save(file_conf_dlas, confidence_test_DLAs)
                    with open('confidence_test_nonDLAs.npy', 'w') as file_conf_nondlas:
                        np.save(file_conf_nondlas, confidence_test_non_DLAs)

            # Log and ignore exceptions
            except Exception as e:
                with open("batch_results.csv", "a") as csvoutput:
                    csvoutput.write(
                        "%d,error,error,error,error,error,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f\n" % \
                        (iteration_num,
                         hyperparameters['learning_rate'], hyperparameters['training_iters'],
                         hyperparameters['batch_size'], hyperparameters['l2_regularization_penalty'],
                         hyperparameters['dropout_keep_prob'], hyperparameters['fc1_n_neurons'],
                         hyperparameters['conv1_kernel'], hyperparameters['conv2_kernel'],
                         hyperparameters['conv1_filters'], hyperparameters['conv2_filters'],
                         hyperparameters['conv1_stride'], hyperparameters['conv2_stride'],
                         hyperparameters['pool1_kernel'], hyperparameters['pool2_kernel'],
                         hyperparameters['pool1_stride'], hyperparameters['pool2_stride'],
                         hyperparameters['nan_parameter'], hyperparameters['pool1_method'],
                         hyperparameters['pool2_method']))

                traceback.print_exc()
