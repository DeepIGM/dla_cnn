# Simple program to resize sprites

from PIL import Image

def resize(layer, num):
    for i in range(num):
        filein = "data/neuron_vis/" + layer + '/' + layer + "_" + str(i) + ".png"
        fileout = "data/neuron_vis/resized/" + layer + "/" + layer + "_" + str(i) + ".png"
        im = Image.open(filein)
        im = im.resize((210,157), Image.ANTIALIAS)
        im.save(fileout)

def main():
    resize("conv1", 100)
    resize("conv1_relu", 100)
    resize("pool1", 100)
    resize("conv2", 96)
    resize("conv2_relu", 96)
    resize("pool2", 96)
    resize("conv3", 96)
    resize("conv3_relu", 96)
    resize("pool3", 96)

if __name__ == "__main__":
    main()
