# This file creates 3D visualizations of our DLA CNN model
import imageio
import os


import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render

import dla_lucid
from dla_lucid import DLA

def get_channel_vis(model, layer, channel, positive=True):
    """
     - Gets the channel visualization of a given layer
     - Optimizes positively by default (when positive=True)
    """
    if positive:
        obj = objectives.channel(dla_lucid.LAYERS[layer][0], channel)
    else:
        obj = - objectives.channel(dla_lucid.LAYERS[layer][0], channel)

    img = render.render_vis(model, obj, dla_lucid.PARAM_3D, thresholds=dla_lucid.THRESH_3D, transforms=dla_lucid.TFORMS_3D, verbose=True)
    return img[0][0]

def get_neuron_vis(model, layer, channel):
    """
    - Get the neuron visualization, given a layer and channel
    """
    obj = objectives.neuron(dla_lucid.LAYERS[layer][0], channel)
    img = render.render_vis(model, obj, dla_lucid.PARAM_3D, thresholds=dla_lucid.THRESH_3D, transforms=[], verbose=False)
    return img[0][0]

def get_unit_vis(model, layer, channel):
    """
    - Gets three different visualizations:
        - One for the positive channel
        - One for the Negative channel
        - One for the neuron objective
    """
    nrn_obj = get_neuron_vis(model, layer, channel)
    pos_channel = get_channel_vis(model, layer, channel)
    neg_channel = get_channel_vis(model, layer, channel, False)
    return nrn_obj, pos_channel, neg_channel

def get_layer_vis(model, layer, all=False):
    """
    - Gets visualizations for all channels in a layer
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
    for layer in dla_lucid.LAYERS:
        create_vis_spritemap(model, layer)
    print("Finished all visualizations.")

if __name__ == "__main__":
    main()




