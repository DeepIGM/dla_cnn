# This file creates spritempas for out 1d Channel Visualizations

from PIL import Image

def create_rows(layer, num_rows, num_cols):
    rows = []
    fn = 0
    for i in range(num_rows):
        row = Image.new('RGBA', ((210*num_cols + 80*num_cols), (257)))
        offset = 40
        for i in range(num_cols):
            image_file = 'data/neuron_vis/resized/' + layer + '/' + layer + '_' + str(fn) + '.png'
            fn += 1

            image = Image.open(image_file)
            row.paste(im=image, box=(offset, 50))
            offset += 290
        rows.append(row)
    return rows


def create_map(layer, num_rows, num_cols):
    map = Image.new('RGBA', ((num_cols*210 + 80*num_cols), (num_rows*257)))
    rows = create_rows(layer, num_rows, num_cols)
    offset = 0
    for i in range(num_rows):
        img = rows[i]
        map.paste(im=img, box=(0, offset))
        offset += 257
    return map

def main():

    map1 = create_map('conv1', 10, 10)
    map2 = create_map('conv1_relu', 10, 10)
    map3 = create_map('pool1', 10, 10)
    map4 = create_map('conv2', 8, 12)
    map5 = create_map('conv2_relu', 8, 12)
    map6 = create_map('pool2', 8, 12)
    map7 = create_map('conv3', 8, 12)
    map8 = create_map('conv3_relu', 8, 12)
    map9 = create_map('pool3', 8, 12)
    #
    #
    map1.save('data/neuron_vis/sprites/conv1.png')
    map2.save('data/neuron_vis/sprites/conv1_relu_resized.png')
    map3.save('data/neuron_vis/sprites/pool1_resized.png')
    map4.save('data/neuron_vis/sprites/conv2_resized.png')
    map5.save('data/neuron_vis/sprites/conv2_relu_resized.png')
    map6.save('data/neuron_vis/sprites/pool2_resized.png')
    map7.save('data/neuron_vis/sprites/conv3_resized.png')
    map8.save('data/neuron_vis/sprites/conv3_relu_resized.png')
    map9.save('data/neuron_vis/sprites/pool3_resized.png')


if __name__ == "__main__":
    main()
