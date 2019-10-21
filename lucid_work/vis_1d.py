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
           'pool3': ['MaxPool_2', 96],
           'fc1': ['MatMul', 350],
           'fc1_relu': ['Relu_3', 350]}

# LAYERS = {'fc2_1': ['MatMul_1', 200],
#           'fc2_1_relu': ['Relu_4', 200],
#           'fc2_2': ['MatMul_2', 350],
#           'fc2_2_relu': ['Relu_5', 350],
#           'fc2_3': ['MatMul_3', 150],
#           'fc2_3_relu': ['Relu_6', 150],
#           'fc_class': ['MatMul_4', 1],
#           'output_class': ['y_nn_classifer', 1],
#           'fc_offset': ['MatMul_5', 1],
#           'fc_offset_out': ['y_nn_offset', 1],
#           'fc_coldens': ['MatMul_6', 1],
#           'fc_coldens_out': ['y_nn_coldensity', 1]}


def vis_channel(model, layer, channel_n):
    print('Getting vis for ' + layer + ', channel ' + str(channel_n))
    l_name = LAYERS[layer][0]
    obj = objectives.channel(l_name, channel_n)
    imgs = render.render_vis(model, obj, PARAM_F, thresholds=THRESH, transforms=TFORMS, verbose=False)
    imgs_array = np.array(imgs)
    imgs_reshaped = imgs_array.reshape(400)
    return imgs_reshaped

def vis_layer(model, layer):
    num_channels = LAYERS[layer][1]
    all_vis = []
    for i in range(num_channels):
        vis = vis_channel(model, layer, i)
        all_vis.append(vis)

    all_vis_array = np.array(all_vis)
    return all_vis_array

def save_layer(model, layer, path):
    vis_array = vis_layer(model, layer)
    outfile = path + layer
    np.save(outfile, vis_array)


def main():
    model = DLA()
    # save_layer(model, 'fc2_1', 'data/')
    # save_layer(model, 'fc2_1_relu', 'data/')
    # save_layer(model, 'fc2_2', 'data/')
    # save_layer(model, 'fc2_2_relu', 'data/')
    # save_layer(model, 'fc2_3', 'data/')
    # save_layer(model, 'fc2_3_relu', 'data/')
    # save_layer(model, 'fc_class', 'data/')
    # save_layer(model, 'output_class', 'data/')
    # save_layer(model, 'fc_offset', 'data/')
    # save_layer(model, 'fc_offset_out', 'data/')
    # save_layer(model, 'fc_coldens', 'data/')
    # save_layer(model, 'fc_coldens_out', 'data/')
    save_layer(model, 'fc1', 'data/')
    save_layer(model, 'fc1_relu', 'data/')



if __name__ == "__main__":
    main()






