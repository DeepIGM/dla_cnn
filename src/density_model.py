#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import random, os, sys, traceback, math, json, timeit
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

    INPUT_SIZE = 400

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
    inputsize_fc3 = int(math.ceil(math.ceil(math.ceil(math.ceil(
        INPUT_SIZE / conv1_stride) / pool1_stride) / conv2_stride) / pool2_stride)) * conv2_filters

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
    cost = tf.reduce_sum(tf.nn.l2_loss(y_nn - y_)) + \
           l2_regularization_penalty * tf.nn.l2_loss(W_conv1) + \
           l2_regularization_penalty * tf.nn.l2_loss(W_conv2) + \
           l2_regularization_penalty * tf.nn.l2_loss(W_fc3) + \
           l2_regularization_penalty * tf.nn.l2_loss(W_fc4)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    prediction = y_nn
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y_nn, y_))))
    return train_step, prediction, rmse, cost, y_, x, keep_prob


def predictions_ann(hyperparameters, flux, labels, checkpoint_filename, TF_DEVICE=''):
    timer = timeit.default_timer()
    BATCH_SIZE = 6000
    n_samples = flux.shape[0]
    pred = np.zeros((1,n_samples), dtype=np.float32)

    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.device(TF_DEVICE), tf.Session() as sess:
        train_step, prediction, rmse, cost, y_, x, keep_prob = build_model(hyperparameters)

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_filename+".ckpt")
        for i in range(0,n_samples,BATCH_SIZE):
            p = sess.run([prediction], feed_dict={x:flux[i:i+BATCH_SIZE,:],
                                                  y_:labels[i:i+BATCH_SIZE], keep_prob: 1.0})
            pred[0,i:i + BATCH_SIZE] = np.array(p)

    print "Density Model processed %d samples in chunks of %d in %0.1f seconds" % \
          (n_samples, BATCH_SIZE, timeit.default_timer() - timer)
    return pred


def train_ann(hyperparameters, save_filename=None, load_filename=None,
              train_npy_file="../data/densitydata_train.npy",
              test_npy_file="../data/densitydata_test.npy"):
    training_iters = hyperparameters['training_iters']
    batch_size = hyperparameters['batch_size']
    dropout_keep_prob = hyperparameters['dropout_keep_prob']

    (train, test) = (DataSet(np.load(train_npy_file)), DataSet(np.load(test_npy_file)))

    # Predefine variables that need to be returned from local scope
    best_rmse = 999999999
    test_rmse = None
    loss_value = None

    with tf.Graph().as_default():
        # Build model
        (train_step, prediction, rmse, cost, y_, x, keep_prob) = build_model(hyperparameters)

        with tf.Session() as sess:
            if load_filename is not None:
                tf.train.Saver().restore(sess, load_filename+".ckpt")
                print("Model loaded from checkpoint: %s"%load_filename)
            else:
                sess.run(tf.initialize_all_variables())

            for i in range(training_iters):
                batch = train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch[0], y_: batch[2], keep_prob: dropout_keep_prob})
                if i % 100 == 0:
                    train_rmse, loss_value = sess.run([rmse, cost], feed_dict={x:batch[0], y_:batch[2], keep_prob: 1.0})
                    #print("step %06d, training rmse/loss      %0.4f / %0.4f" % (i, train_rmse/batch_size, np.mean(loss_value)))
                    print("step %06d, training rmse/loss      %0.4f / %0.4f" % (i, train_rmse, np.mean(loss_value)))
                if i % 1000 == 0 or i == training_iters - 1:
                    test_rmse = sess.run(rmse, feed_dict={x:test.fluxes[0::4], y_:test.col_density[0::4], keep_prob:1.0})
                    #test_rmse = test_rmse / math.ceil(float(np.shape(test.fluxes)[0])/float(4))
                    best_rmse = test_rmse if (test_rmse < best_rmse) else best_rmse
                    print("             test RMSE:     %0.4f" % test_rmse)

            # Save checkpoint
            if save_filename is not None:
                saver = tf.train.Saver()
                save_path = saver.save(sess, save_filename + ".ckpt")
                with open(save_filename + "_hyperparams.json", 'w') as fp:
                    json.dump(hyperparameters, fp)
                print("Model saved in file: %s" % save_path)

    return best_rmse, test_rmse, np.mean(loss_value)


if __name__ == '__main__':
    #
    # Execute batch mode
    #
    RUN_SINGLE_ITERATION = False
    checkpoint_filename = "../models/density_model"
    load_filename = None #"../models/density_model"       # Set to None to begin from scratch

    iteration_num = 0
    parameter_names = ["learning_rate", "training_iters", "batch_size", "l2_regularization_penalty", "dropout_keep_prob",
                       "fc1_n_neurons", "conv1_kernel", "conv2_kernel", "conv1_filters", "conv2_filters",
                       "conv1_stride", "conv2_stride", "pool1_kernel", "pool2_kernel", "pool1_stride", "pool2_stride",
                       "nan_parameter", "pool1_method", "pool2_method"]
    parameters = [
        # First column: Keeps the best best parameter based on accuracy score
        # Other columns contain the parameter options to try

        # learning_rate
        [0.001,         0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        # [0.00005],
        # training_iters
        [20000],
        # [10000],
        # batch_size
        [150,           100, 150, 200, 300, 500],
        # l2_regularization_penalty
        [0.005,         0.01, 0.008, 0.005, 0.003, 0.001],
        # dropout_keep_prob
        [1,             0.8, 0.9, 0.95, 0.98, 1],
        # fc1_n_neurons
        [500,           100, 300, 500, 750, 1000],
        # conv1_kernel
        [18,            14, 16, 17, 18, 19, 20, 22, 28, 32],
        # conv2_kernel
        [16,            8, 12, 16, 18, 20, 24, 28, 32],
        # conv1_filters
        [80,            48, 64, 80, 90, 100, 120],
        # conv2_filters
        [96,            64, 80, 96, 128, 150],
        # conv1_stride
        [3,             1, 2, 3, 4, 5],
        # conv2_stride
        [1,             1, 2, 3, 4],
        # pool1_kernel
        [4,             2, 3, 4, 5, 6],
        # pool2_kernel
        [7,             4, 5, 6, 7, 8, 9, 10, 11, 12],
        # pool1_stride
        [2,             1, 2, 3, 4],
        # pool2_stride
        [2,             1, 2, 3, 4],
        # nan_parameter
        [0.0,           ],
        # pool1_method
        [1,             1, 2],
        #pool2_method
        [1,             1, 2]
    ]

    # Write out CSV header
    batch_results_file = '../tmp/batch_results.csv'
    os.remove(batch_results_file) if os.path.exists(batch_results_file) else None
    with open(batch_results_file, "a") as csvoutput:
        csvoutput.write("iteration_num,best_rmse,last_rmse,last_objective," + ",".join(parameter_names) + "\n")

    while 1:
        if RUN_SINGLE_ITERATION and iteration_num >= 1:
            exit()
        iteration_best_rmse = 99999999
        i = random.randint(0,len(parameters)-1)           # Choose a random parameter to change
        hyperparameters = {}

        for j in (range(1,len(parameters[i])) if not RUN_SINGLE_ITERATION else range(0,1)):
            iteration_num += 1

            for k in range(0, len(parameter_names)):
                hyperparameters[parameter_names[k]] = parameters[k][j] if i == k else parameters[k][0]

            try:
                # Print parameters during training
                print("----------------------------------------------------------------------------------------------")
                print("PARAMETERS (varying %s): " % parameter_names[i])
                for k in range(0,len(parameters)):
                    sys.stdout.write("{:<30}{:<15}\n".format( parameter_names[k], parameters[k][j] if k==i else parameters[k][0] ))

                # ANN Training
                (best_rmse, last_rmse, last_objective) = \
                    train_ann(hyperparameters, save_filename=checkpoint_filename, load_filename=load_filename)

                # Save results and parameters to CSV
                with open("batch_results.csv", "a") as csvoutput:
                    csvoutput.write("%d,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f\n" % \
                                    (iteration_num, best_rmse, last_rmse, last_objective,
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
                if best_rmse <= iteration_best_rmse:
                    iteration_best_rmse = best_rmse
                    parameters[i][0] = parameters[i][j]
                    print("Best RMSE for parameter [%s] with RMSE [%0.4f] now set to [%f]" % (parameter_names[i], iteration_best_rmse, parameters[i][0]))

                # if not RUN_SINGLE_ITERATION:
                #     # Predictions
                #     p_data = DataSet(np.load("densitydata_test.npy"))
                #     predictions_test = predictions_ann(hyperparameters, p_data.fluxes[0:10000, :],
                #                                        p_data.col_density[0:10000], checkpoint_filename)
                #     print(np.shape(predictions_test))
                #     print(predictions_test[0:200])
                #
                #     with open('densitypredictions_test.npy', 'w') as file_pred_test:
                #         np.save(file_pred_test, predictions_test)

            # Log and ignore exceptions
            except Exception as e:
                with open("batch_results.csv", "a") as csvoutput:
                    csvoutput.write("%d,error,error,error,error,error,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f,%07f\n" % \
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


