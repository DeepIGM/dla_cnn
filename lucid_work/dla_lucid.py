"""
This is our Lucid model for our DLA CNN
"""

from lucid.modelzoo.vision_base import Model
import lucid.optvis.param as param
import lucid.optvis.transform as transform

"""
LAYERS is a dictionary that contains information about the layers in our model
We simplify the names of each layer in order to make things easier to read.

conv1: [Conv2D, 100]

'conv1' is how we refer to this layer
'Conv2D' is the actual name of the graddef node
100 is the number of channles in the layer
"""
LAYERS = { 'conv1': ['Conv2D', 100],
           'conv1_relu': ['Relu', 100],
           'pool1': ['MaxPool', 100],
           'conv2': ['Conv2D_1', 96],
           'conv2_relu': ['Relu_1', 96],
           'pool2': ['MaxPool_1', 96],
           'conv3': ['Conv2D_2', 96],
           'conv3_relu': ['Relu_2', 96],
           'pool3': ['MaxPool_2', 96]}

"""
Lucid model definition for our DLA CNN
"""
class DLA(Model):
    model_path = 'protobufs/full_model_8_13.pb'
    image_shape = [1, 400]
    image_value_range = [0,1]
    input_name = 'x'


# Parameritization, thresholds, and transforms for creating 1D optimizations
PARAM_1D = lambda: param.image(400, h=1, channels=1)
THRESH_1D = (256,)
TFORMS_1D = []

# Parameritization, thresholds, and transforms for creating 3D optimizations
PARAM_3D = lambda: param.image(116, h=116, alpha=False)
THRESH_3D = (256,)
TFORMS_3D = [transform.pad(4), transform.jitter(4)]

