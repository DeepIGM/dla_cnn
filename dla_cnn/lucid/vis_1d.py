# This file creates 1D visualizations for our model
# Can create channel or neuron visualizations

import numpy as np

import lucid.optvis.objectives as objectives
import lucid.optvis.render as render

from dla_cnn.lucid import dla_lucid
from dla_cnn.lucid.dla_lucid import DLA


def vis_channel(model, layer, channel_n):
    """

    This function creates a visualization for a single channel in a layer

    :param model: model we are visualizing
    :type model: lucid.modelzoo

    :param layer: the name of the layer we are visualizing
    :type layer: string

    :param channel_n: The channel number in the layer we are optimizing for
    :type channel_n: int

    :return: array of pixel values for the visualization
    """
    print('Getting vis for ' + layer + ', channel ' + str(channel_n))
    l_name = dla_lucid.LAYERS[layer][0]
    obj = objectives.channel(l_name, channel_n)
    imgs = render.render_vis(model, obj, dla_lucid.PARAM_1D,
                             thresholds=dla_lucid.THRESH_1D,
                             transforms=dla_lucid.TFORMS_1D, verbose=False)
                             #use_fixed_seed=True)  # This may not be function for TF 1
    imgs_array = np.array(imgs)
    imgs_reshaped = imgs_array.reshape(400)
    return imgs_reshaped

def vis_neuron(model, layer, channel_n):
    """

    This function creates a visualization for a single neuron in a layer
    The neuron objective defaults to the center neuron in the channel

    :param model: model we are visualizing
    :type model: lucid.modelzoo

    :param layer: the name of the layer we are visualizing
    :type layer: string

    :param channel_n: The channel number in the layer we are optimizing for
    :type channel_n: int

    :return: array of pixel values for the visualization
    """

    print('getting vis for ' + layer + ', channel ' + str(channel_n))
    l_name = dla_lucid.LAYERS[layer][0]
    obj = objectives.neuron(l_name, channel_n)
    imgs = render.render_vis(model, obj, dla_lucid.PARAM_1D,
                             thresholds=dla_lucid.THRESH_1D, transforms=dla_lucid.TFORMS_1D, verbose=False)
    imgs_array = np.array(imgs)
    imgs_reshaped = imgs_array.reshape(400)
    return imgs_reshaped


def vis_layer(model, layer, channel):
    """
    This function creates visualizations for an entire layer

    :param model: model we are visualization
    :type model" lucid.modelzoo

    :param layer: the name of the layer we are optimizing for
    :type layer: string

    :param channel: True for creating channel vis, False for creating neuron vis
    :type channel: boolean

    :return: array of all pixel values in the layers visualizations
    """
    num_channels = dla_lucid.LAYERS[layer][1]
    all_vis = []
    for i in range(num_channels):
        if channel is True:
            vis = vis_channel(model, layer, i)
        else:
            vis = vis_neuron(model, layer, i)
        all_vis.append(vis)

    all_vis_array = np.array(all_vis)
    return all_vis_array

def save_layer(model, layer, path, channel):

    """
    This function calles vis_layer() to create layer visualizations,
    and then saves to a folder

    :param model: model we are optimizing for
    :type model: lucid.modelzoo

    :param layer: the name of the layer we are visualizing
    :type layer: string

    :param path: path to save visualizations too, must already exist
    :type path: string

    :param channel: True for creating channel vis, False for creating neuron vis
    :type channel: boolean

    :return: nothing
    """

    # If channel is true, create channel vis
    # else create neuron vis
    vis_array = vis_layer(model, layer, channel)
    outfile = path + layer
    np.save(outfile, vis_array)


def main():
    model = DLA()

    # Neuron Vis of the model
    save_layer(model, 'conv1', 'data/', False)
    save_layer(model, 'conv1_relu', 'data/', False)
    save_layer(model, 'pool1', 'data/', False)
    save_layer(model, 'conv2', 'data/', False)
    save_layer(model, 'conv2_relu', 'data/', False)
    save_layer(model, 'pool2', 'data/', False)
    save_layer(model, 'conv3', 'data/', False)
    save_layer(model, 'conv3_relu', 'data/', False)
    save_layer(model, 'pool3', 'data/', False)

    # Channel Vis of the model
    save_layer(model, 'conv1', 'data/', True)
    save_layer(model, 'conv1_relu', 'data/', True)
    save_layer(model, 'pool1', 'data/', True)
    save_layer(model, 'conv2', 'data/', True)
    save_layer(model, 'conv2_relu', 'data/', True)
    save_layer(model, 'pool2', 'data/', True)
    save_layer(model, 'conv3', 'data/', True)
    save_layer(model, 'conv3_relu', 'data/', True)
    save_layer(model, 'pool3', 'data/', True)

if __name__ == "__main__":
    main()






