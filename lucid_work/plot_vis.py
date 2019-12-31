# This file plots the saved visualizations
# Visulaoizations were saved as '.npy' files, this file
# plots and saves as '.png' files

import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def save_vis(layer, vis, channel_n):
    """

    This file takes an array of pixel values from an individual visualization,
    plots it, and saves the image to a png file

    :param layer: layer we are saving
    :type layer: string

    :param vis: array of pixel values for an individual visualization to be plotted
    :type vis: array

    :param channel_n: channel of the visualization we are saving
    :type channel_n: int

    :return: nothing
    """


    fig = plt.figure(frameon=False);
    ax = plt.Axes(fig, [0, 0, 1, 1]);
    ax.set_axis_off();
    fig.add_axes(ax);
    ax.plot(vis, 'black');
    ax.set(xlim=(0, 400));
    file_save = 'data/neuron_vis/' + layer + '/' + layer + '_' + str(channel_n) +'.png'
    fig.savefig(file_save);
    plt.close(fig)


def create_layer_vis(filein, layer):
    """
    This file loads in visualization array of pixels and call save_vis() in
    order to plot each visualization separately.

    :param filein: path of the file that has layer visualization data
    :type filein: string

    :param layer: the layer we are saving visualizations for
    :type layer: string
    :return: nothing
    """

    imgs = np.load(filein)

    for i in range(len(imgs)):
        save_vis(layer, imgs[i], i)

def main():

    create_layer_vis('data/conv1.npy', 'conv1')
    create_layer_vis('data/conv1_relu.npy', 'conv1_relu')
    create_layer_vis('data/pool1.npy', 'pool1')
    create_layer_vis('data/conv2.npy', 'conv2')
    create_layer_vis('data/conv2_relu.npy', 'conv2_relu')
    create_layer_vis('data/pool2.npy', 'pool2')
    create_layer_vis('data/conv3.npy', 'conv3')
    create_layer_vis('data/conv3_relu.npy', 'conv3_relu')
    create_layer_vis('data/pool3.npy', 'pool3')


if __name__ == "__main__":
    main()
