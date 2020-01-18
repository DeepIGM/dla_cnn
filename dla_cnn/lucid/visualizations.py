# This file creates 3D visualizations of our DLA CNN model
import imageio
import os


import lucid.optvis.objectives as objectives
import lucid.optvis.render as render

import dla_lucid
from dla_lucid import DLA

def get_channel_vis(model, layer, channel, positive=True):
    """
     This function creates a single channel optimzation from a given layer and channel

    :param model: Our lucid model
    :type model: lucid.modelzoo

    :param layer: The layer we are visualizing
    :type layer: string

    :param channel: The channel we optimizing fot
    :type: int

    :param positive: True for positive optimization, False for negative optimization
    :type positive: boolean

    :return: the image optimzation
    """
    if positive:
        obj = objectives.channel(dla_lucid.LAYERS[layer][0], channel)
    else:
        obj = - objectives.channel(dla_lucid.LAYERS[layer][0], channel)

    img = render.render_vis(model, obj, dla_lucid.PARAM_3D, thresholds=dla_lucid.THRESH_3D, transforms=dla_lucid.TFORMS_3D, verbose=True)
    return img[0][0]

def get_neuron_vis(model, layer, channel):
    """
    This function creates visualizations for a neuron objective

    :param model: our model we are visualizing
    :type model: lucid.modelzoo

    :param layer: the layer we are interested in
    :type layer: string

    :param channel: the channel we are optimizing for. The neuron objective defaults to center neuron in the channel
    :type channel: int

    :return: the image optimization
    """
    obj = objectives.neuron(dla_lucid.LAYERS[layer][0], channel)
    img = render.render_vis(model, obj, dla_lucid.PARAM_3D, thresholds=dla_lucid.THRESH_3D, transforms=[], verbose=False)
    return img[0][0]

def get_unit_vis(model, layer, channel):
    """
    This function creates a positive and negative channel visualization
    as well as a neuron visualization for a single channel

    :param model: our model we are visualizing
    :type model: lucid.modelzoo

    :param layer: the layer we are interested in
    :type layer: string

    :param channel: the channel we are optimizing for
    :type channel: int

    :return: three image optimizations

    """
    nrn_obj = get_neuron_vis(model, layer, channel)
    pos_channel = get_channel_vis(model, layer, channel)
    neg_channel = get_channel_vis(model, layer, channel, False)
    return nrn_obj, pos_channel, neg_channel

def get_layer_vis(model, layer, all=False):
    """
    This function creates calls either get_channel_vis() or
    get_unit_vis() to create visualizations for evey single channel
    in a layer.

    :param model: our model we are visualizing
    :type model: lucid.modelzoo

    :param layer: the layer we are visualizing
    :type layer: string

    :param all: False to only create normal positive optimizations, True
    to create Positive, negative, and neuron optimizations

    :return: all images optimizations from the layer
    """
    num_channels = dla_lucid.LAYERS[layer][1]
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
            vis = get_channel_vis(model, layer, i)
            img.append(vis)
        imgs.append(img)
    return imgs

def save_vis_all(imgs, outdir):
    """
    Function to save positive, negative and neuron objectives

    :param imgs: list of images, three for each channel
    :type imgs: list

    :param outdir: the path to save visualizations to
    :type outdir: string

    :return: none

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
    This function calls get_layer_vis() on each layer in order to
    create visualizations for all channels in pooling layers. It then
    saves all by calling save_vis_all
    :param model: our model we are visualizing
    :return: none
    """
    for layer in dla_lucid.LAYERS:
        save_dir = 'visualizations/' + layer + '/'
        imgs = get_layer_vis(model, layer, True)
        save_vis_all(imgs, save_dir)
        print("Saved all visualizations from " + layer)

def save_spritemaps(layer, imgs, dir):
    """
    This function saves images from a layer to a directory

    :param layer: the layer we are saving visualizations for
    :param imgs: the list of images to save
    :param dir: directory to save images too
    :return: nothing
    """
    for i in range(len(imgs)):
        try:
            os.makedirs(dir, 0o777)
        except:
            pass
        file = layer + '_' + str(i) + '.png'
        vis = imgs[i][0]
        imageio.imwrite(dir+file, vis)

def create_vis_spritemap(model, layer):
    """
    Creates all visualizations from a layer and saves them to a directory
    :param model: our model we are visualizing
    :param layer: the layer we are interested in
    :return: nothing
    """
    dir = 'spritemap_vis/' + layer +'/'
    imgs = get_layer_vis(model, layer, False)
    save_spritemaps(layer, imgs, dir)
    print("Saved all visualizations from " + layer)


def main():
    model = DLA()

    # To create normal positive channel visualizations
    for layer in dla_lucid.LAYERS:
        print(layer)
        create_vis_spritemap(model, layer)

    print("Finished all visualizations.")


if __name__ == "__main__":
    main()




