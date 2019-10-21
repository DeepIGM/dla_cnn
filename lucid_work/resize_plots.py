from PIL import Image



def resize(layer, num):
    for i in range(num):
        filein = "data/model_1d_vis/" + layer + "/" + layer + "_" + str(i) + ".png"
        fileout = "data/model_1d_vis/" + layer + "/resized/" + layer + "_" + str(i) + ".png"
        im = Image.open(filein)
        im = im.resize((210,157), Image.ANTIALIAS)
        im.save(fileout)

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


def main():
    resize("fc1", 350)
    resize("fc1_relu", 350)
    resize("fc2_1", 200)
    resize("fc2_1_relu", 200)
    resize("fc2_2", 350)
    resize("fc2_2_relu", 350)
    resize("fc2_3", 150)
    resize("fc2_3_relu", 150)

if __name__ == "__main__":
    main()
