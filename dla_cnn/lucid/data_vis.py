""" Generate visualizations related to input data"""
import numpy as np
import tensorflow as tf  #  Needs to be < 2.0

import lucid.optvis.render as render

from dla_cnn.lucid import dla_lucid

from IPython import embed

def dla_semantic_dict(data, layer):

    # Reshape for Lucid
    flux_test = data.reshape(1, 400, 1, 1)

    model = dla_lucid.DLA()
    model_layer = dla_lucid.LAYERS[layer][0]

    input_1d = flux_test  # Actual 1-Dimensional test input
    #     print(input_1d)
    #img = load(img_file)

    # Compute the activations
    with tf.Graph().as_default(), tf.Session() as sess:
        t_input = tf.placeholder(tf.float32, shape=[1, 400, 1, 1])
        T = render.import_model(model, t_input, t_input)
        acts = T(model_layer).eval(feed_dict={t_input: input_1d,
                                              'import/keep_prob:0': .98})[0]

    # Find best position for our initial view
    max_mag = acts.max(-1)
    max_x = np.argmax(max_mag.max(-1))

    '''
    # Find appropriate spritemap
    spritemap_n, spritemap_url = dla_spritemap(layer)
    if hide == False:
        # Display the interactive display by calling the svelte component
        # Actually construct the semantic dictionary interface
        # using our *custom component*
        lucid_svelte.SemanticDict({
            "spritemap_url": spritemap_url,
            "sprite_size": 210,
            "sprite_n_wrap": spritemap_n,
            "image_url": _image_url(img),
            "activations": [
                [[{"n": float(n), "v": float(act_vec[n])} for n in np.argsort(-act_vec)[:4]] for act_vec in
                 act_slice] for act_slice in acts],
            "pos": [float(max_x), 0],
            "denom": 150

        })
    '''
    return acts


def get_max_vals(activations):
    max_acts = np.max(activations, axis=2).flatten()
    imax = np.argmax(activations, axis=2)
    '''
    max_acts = []
    for a in activations:
        m = max(a[0])
        max_acts.append(m)
    '''
    return max_acts, imax


