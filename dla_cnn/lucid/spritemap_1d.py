# This file creates spritempas for out 1d Channel Visualizations

from PIL import Image

def create_rows(layer, num_rows, num_cols, sprite_width, sprite_height, row_spacing, col_spacing):
    """
    This function takes in individual channel visualizations from a layer and puts them together
    in a single image spritemap row. Spacing is added between each visualization so they dont appear
    stacked on top of each other.

    :param layer: The layer we are creating our spritemap for
    :param num_rows: The number of rows in the spritemap
    :param num_cols: The number of columns in the spritemap
    :param sprite_width: The width (in pixels) of each individual visualization
    :param sprite_height: The height (in pixels) of each individual visualization
    :param row_spacing: The amount of spacing (in pixels) between each visualization in a row
    :param col_spacing: The amount of spacing (in pixels) between each visualization in a column
    :return: List of PIL images, one for each row in the spritemap
    """

    row_width = (num_cols * sprite_width) + (row_spacing * num_cols)
    row_height = sprite_height + col_spacing

    rows = []
    fn = 0
    for i in range(num_rows):
        row = Image.new('RGBA', (row_width,row_height))
        offset = row_spacing / 2
        for i in range(num_cols):
            image_file = 'data/neuron_vis/resized/' + layer + '/' + layer + '_' + str(fn) + '.png'
            fn += 1

            image = Image.open(image_file)
            row.paste(im=image, box=(offset, (col_spacing/2)))
            offset += (sprite_width + row_spacing)
        rows.append(row)
    return rows


def create_map(layer, num_rows, num_cols, sprite_width, sprite_height, row_spacing, col_spacing):
    """
    This function creates a full sized map and adds rows created from the create_rows() function

    :param layer: Layer we are creating spritemap for
    :param num_rows: Number of rows in spritemap
    :param num_cols: Number of cols in spritemap
    :param sprite_width: The width (pixels) of each individual visualization
    :param sprite_height: The height (pixels) of each individual visualization
    :param row_spacing: Spacing (pixels) between visualizations in a row
    :param col_spacing: Spacing (pixels) between visualizations in a column
    :return: Full image spritemap
    """
    row_width = (num_cols*sprite_width) + (row_spacing*num_cols)
    row_height = sprite_height + col_spacing

    # create blank map
    map = Image.new('RGBA', (row_width, (num_rows*row_height)))

    # Get images for each row
    rows = create_rows(layer, num_rows, num_cols,sprite_width, sprite_height, row_spacing, col_spacing)
    offset = 0

    for i in range(num_rows):
        # past row images into map
        img = rows[i]
        map.paste(im=img, box=(0, offset))
        offset += row_height
    return map

def main():

    map1 = create_map('conv1', 10, 10, 210, 157, 80, 100)
    map2 = create_map('conv1_relu', 10, 10, 210, 157, 80, 100)
    map3 = create_map('pool1', 10, 10, 210, 157, 80, 100)
    map4 = create_map('conv2', 8, 12, 210, 157, 80, 10)
    map5 = create_map('conv2_relu', 8, 12, 210, 157, 80, 10)
    map6 = create_map('pool2', 8, 12, 210, 157, 80, 10)
    map7 = create_map('conv3', 8, 12, 210, 157, 80, 10)
    map8 = create_map('conv3_relu', 8, 12, 210, 157, 80, 10)
    map9 = create_map('pool3', 8, 12, 210, 157, 80, 10)
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
