# This file plots the saved visualizations
# Visulaoizations were saved as '.npy' files, this file
# plots and saves as '.png' files

import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def save_vis(layer, vis, channel_n):
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
