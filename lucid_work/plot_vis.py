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
    file_save = 'data/model_1d_vis/' + layer + '/' + layer + '_' + str(channel_n) +'.png'
    fig.savefig(file_save);
    plt.close(fig)


def create_layer_vis(layer):
    data = 'data/model_1d_vis/' + layer + '/' + layer + '.npy'
    imgs = np.load(data)

    for i in range(len(imgs)):
        save_vis(layer, imgs[i], i)

def main():
    # create_layer_vis('conv1')
    # create_layer_vis('conv1_relu')
    # create_layer_vis('pool1')
    # create_layer_vis('conv2')
    # create_layer_vis('conv2_relu')
    # create_layer_vis('pool2')
    # create_layer_vis('conv3')
    # create_layer_vis('conv3_relu')
    # create_layer_vis('pool3')
    create_layer_vis('fc2_1')
    create_layer_vis('fc2_1_relu')
    create_layer_vis('fc2_2')
    create_layer_vis('fc2_2_relu')
    create_layer_vis('fc2_3')
    create_layer_vis('fc2_3_relu')
    create_layer_vis('fc_class')
    create_layer_vis('output_class')
    create_layer_vis('fc_offset')
    create_layer_vis('fc_offset_out')
    create_layer_vis('fc_coldens')
    create_layer_vis('fc_coldens_out')
    create_layer_vis('fc1')
    create_layer_vis('fc1_relu')



if __name__ == "__main__":
    main()
