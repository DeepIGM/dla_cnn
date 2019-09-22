import numpy as np
import tensorflow as tf
import scipy.ndimage as nd
import time
import datetime
import imageio
import os
import errno

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
from lucid.modelzoo.vision_base import Model

THRESHOLDS = (256,)
TRANSFORMS = [transform.pad(4), transform.jitter(4)]
#PARAM_F = param.image(100)

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

OUTPUT_LAYERS = {'fc2_1': ['MatMul_1', 200],
                 'fc2_1_relu': ['Relu_4', 200],
                 'fc2_2': ['MatMul_2', 350],
                 'fc2_2_relu': ['Relu_5', 350],
                 'fc2_3': ['MatMul_3', 150],
                 'fc2_3_relu': ['Relu_6', 150],
                 'fc_class': ['MatMul_4', 1],
                 'output_class': ['y_nn_classifer', 75],
                 'fc_offset': ['MatMul_5', 1],
                 'fc_offset_out': ['y_nn_offset', 75],
                 'fc_coldens': ['MatMul_6', 1],
                 'fc_coldens_out': ['y_nn_coldensity', 75]}


class DLA(Model):
    model_path = 'protobufs/full_model_8_13.pb'
    image_shape = [1, 400]
    image_value_range = [0,1]
    input_name = 'x'


def get_channel_vis(model, param_f, layer, channel, positive=True):
    """
     - Gets the channel visualization of a given layer
     - Optimizes positively by default (when positive=True)
    """
    if positive:
        obj = objectives.channel(OUTPUT_LAYERS[layer][0], channel)
    else:
        obj = - objectives.channel(OUTPUT_LAYERS[layer][0], channel)
    img = render.render_vis(model, obj, param_f, thresholds=THRESHOLDS, transforms=TRANSFORMS, verbose=True)
    return img[0][0]

def get_neuron_vis(model, param_f, layer, channel):
    """
    - Get the neuron visualization, given a layer and channel
    """
    obj = objectives.neuron(layer, channel)
    img = render.render_vis(model, obj, param_f, thresholds=THRESHOLDS, transforms=[], verbose=False)
    return img[0][0]

def get_unit_vis(model, layer, channel):
    """
    - Gets three different visualizations:
        - One for the positive channel
        - One for the Negative channel
        - One for the neuron objective
    """
    nrn_obj = get_neuron_vis(model, PARAM_F, OUTPUT_LAYERS[layer][0], channel)
    pos_channel = get_channel_vis(model, PARAM_F, OUTPUT_LAYERS[layer][0], channel)
    neg_channel = get_channel_vis(model, PARAM_F, OUTPUT_LAYERS[layer][0], channel, False)
    return nrn_obj, pos_channel, neg_channel

def get_layer_vis(model, layer, all=False):
    """
    - Gets visualizations for all channels in a layer
    """
    num_channels = OUTPUT_LAYERS[layer][1]
    imgs = []
    for i in range(num_channels):
        print("Getting vis for layer: " + layer + ", channel: " + str(i))
        img = []
        if all:
            nrn, pos, neg = get_unit_vis(model, layer, i)
            img.append(nrn)
            img.append(pos)
            img.append(neg)
        else:
            param_reg = lambda: param.image(116, h=116, alpha=False)
            param_alpha = lambda: param.image(116, h=116, alpha=True)
            reg_vis = get_channel_vis(model, param_reg, layer, i)
            alp_vis = get_channel_vis(model, param_alpha, layer, i)
            img.append(reg_vis)
            img.append(alp_vis)
        imgs.append(img)
    return imgs

def save_vis_all(layer, imgs, outdir):
    """
    - saves visualizatiohns
    """
    for i in range(len(imgs)):
        channel_dir = outdir + 'channel_' + str(i) + '/'
        os.makedirs(channel_dir, 0o777)
        neuron = 'neuron_obj_' + str(i) + '.png'
        positive = 'pos_channel_' + str(i) + '.png'
        negative = 'neg_channel_' + str(i) + '.png'
        imageio.imwrite(channel_dir+neuron, imgs[i][0])
        imageio.imwrite(channel_dir+positive, imgs[i][1])
        imageio.imwrite(channel_dir+negative, imgs[i][2])

def create_all_vis(model):
    """
    - Create visualizations for all convolution and pooling layers
    """
    for layer in LAYERS:
        save_dir = 'visualizations/' + layer + '/'
        imgs = get_layer_vis(model, layer)
        save_vis_all(layer, imgs, save_dir)
        print("Saved all visualizations from " + layer)

# def create_all_vis_fc1(model):
#     """
#     - Creates Visualizations for first Fully connected layers
#     """
#     for layer in OUTPUT_LAYERS:
#         if layer == 'fc1' or layer == 'fc1_relu':
#             save_dir = 'visualizations/' + layer + '/'
#             imgs = get_layer_vis(model, layer)
#             save_vis(layer, imgs, save_dir)
#             print("Saved all visualizations from " + layer)

def save_spritemaps(layer, imgs, regdir, alphadir):
    for i in range(len(imgs)):
        try:
            os.makedirs(regdir, 0o777)
            os.makedirs(alphadir, 0o777)
        except:
            pass
        reg_file = layer + '_' + str(i) + '.png'
        alpha_file = layer + '_alpha_' + str(i) + '.png'
        reg_vis = imgs[i][0]
        alpha_vis = imgs[i][1]
        imageio.imwrite(regdir+reg_file, reg_vis)
        imageio.imwrite(alphadir+alpha_file, alpha_vis)

def create_vis_spritemap(model, layer):
   reg_dir = 'spritemap_vis/' + layer +'/'
   alpha_dir = 'spritemap_vis/' + layer + '_alpha/'
   imgs = get_layer_vis(model, layer, False)
   save_spritemaps(layer, imgs, reg_dir, alpha_dir)
   print("Saved all visualizations from " + layer)


def main():
    model = DLA()
    #create_vis_spritemap(model, 'conv1')
    for layer in OUTPUT_LAYERS:
        if layer != 'fc2_1' or layer != 'fc2_1_relu':
            create_vis_spritemap(model, layer)

    print("Finished all visualizations.")


if __name__ == "__main__":
    main()




