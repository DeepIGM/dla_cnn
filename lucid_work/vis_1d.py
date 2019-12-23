# This file creates 1D visualizations for our model
# Can create channel or neuron visualizwtions

import numpy as np
import tensorflow as tf
import scipy.ndimage as nd
import time
import imageio

import matplotlib
import matplotlib.pyplot as plt

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

from lucid.modelzoo.vision_base import Model

class DLA(Model):
    model_path = 'protobufs/full_model_8_13.pb'
    image_shape = [1, 400]
    image_value_range = [0,1]
    input_name = 'x'

PARAM_F = lambda: param.image(400, h=1, channels=1)
THRESH = (256,)
TFORMS = []


LAYERS = { 'conv1': ['Conv2D', 100],
           'conv1_relu': ['Relu', 100],
           'pool1': ['MaxPool', 100],
           'conv2': ['Conv2D_1', 96],
           'conv2_relu': ['Relu_1', 96],
           'pool2': ['MaxPool_1', 96],
           'conv3': ['Conv2D_2', 96],
           'conv3_relu': ['Relu_2', 96],
           'pool3': ['MaxPool_2', 96]}

def vis_channel(model, layer, channel_n):
    print('Getting vis for ' + layer + ', channel ' + str(channel_n))
    l_name = LAYERS[layer][0]
    obj = objectives.channel(l_name, channel_n)
    imgs = render.render_vis(model, obj, PARAM_F, thresholds=THRESH, transforms=TFORMS, verbose=False)
    imgs_array = np.array(imgs)
    imgs_reshaped = imgs_array.reshape(400)
    return imgs_reshaped

def vis_neuron(model, layer, channel_n):
    print('getting vis for ' + layer + ', channel ' + str(channel_n))
    l_name = LAYERS[layer][0]
    obj = objectives.neuron(l_name, channel_n)
    imgs = render.render_vis(model, obj, PARAM_F, thresholds=THRESH, transforms=TFORMS, verbose=False)
    imgs_array = np.array(imgs)
    imgs_reshaped = imgs_array.reshape(400)
    return imgs_reshaped


def vis_layer(model, layer, channel):
    num_channels = LAYERS[layer][1]
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

if __name__ == "__main__":
    main()






