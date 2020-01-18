# THis file creates 3d spritempaps from our 3d visualizations

from PIL import Image

def create_rows(layer, num_rows, num_cols, vis_height, vis_width, alpha=False):
    """
    THis function creates rows for 3D spritemaps

    :param layer: The layer we are creating spritempa for
    :param num_rows: number of rows in spritemap
    :param num_cols: number of columns in spritemap
    :param vis_height: Width (pixels) of individual visualizations
    :param vis_width: Height (pixels) of individual visualizations
    :param alpha: True is want alpha vis, False for normal
    :return: Array or row PIL images
    """
    rows = []
    fn = 0
    for i in range(num_rows):
        row = Image.new('RGB', ((vis_width*num_cols), (vis_height)))
        offset = 0
        for i in range(num_cols):
            if alpha:
                image_file = 'spritemap_vis/' + layer + '_alpha/' + layer + '_alpha_' + str(fn) + '.png'
                fn += 1
            else:
                image_file = 'spritemap_vis/' + layer + '/' + layer + '_' + str(fn) + '.png'
                fn += 1

            image = Image.open(image_file)
            row.paste(im=image, box=(offset, 0))
            offset += vis_width
        rows.append(row)
    return rows


def create_map(layer, num_rows, num_cols, vis_height, vis_width, alpha=False):
    """
    This functions pieces together rows into one single spritemap

    :param layer: layer we are creating map for
    :param num_rows: number of rows in spritemap
    :param num_cols: number of cols in spritemap
    :param vis_height: height(pixels) of an individual visualization
    :param vis_width: width(pixels) of an individual visualization
    :param alpha: True is want alpha vis, False for normal
    :return: Entire spritemap
    """
    map = Image.new('RGB', ((num_cols*vis_width), (num_rows*vis_height)))
    rows = create_rows(layer, num_rows, num_cols, vis_height, vis_width, alpha)
    offset = 0
    for i in range(num_rows):
        img = rows[i]
        map.paste(im=img, box=(0, offset))
        offset += vis_height
    return map

def main():


    map1  = create_map('conv1', 10, 10, False)
    map2  = create_map('conv1', 10, 10, True)
    map3  = create_map('conv1_relu', 10, 10, False)
    map4  = create_map('conv1_relu', 10, 10, True)
    map5  = create_map('pool1', 10, 10, False)
    map6  = create_map('pool1', 10, 10, True)
    map7  = create_map('conv2', 8, 12, False)
    map8  = create_map('conv2', 8, 12, True)
    map9  = create_map('conv2_relu', 8, 12, False)
    map10 = create_map('conv2_relu', 8, 12, True)
    map11 = create_map('pool2', 8, 12, False)
    map12 = create_map('pool2', 8, 12, True)
    map13 = create_map('conv3', 8, 12, False)
    map14 = create_map('conv3', 8, 12, True)
    map15 = create_map('conv3_relu', 8, 12, False)
    map16 = create_map('conv3_relu', 8, 12, True)
    map17 = create_map('pool3', 8, 12, False)
    map18 = create_map('pool3', 8, 12, True)

    map1.save('spritemaps/spritemap_conv1.jpeg', 'jpeg')
    map2.save('spritemaps/spritemap_conv1_alpha.jpeg', 'jpeg')
    map3.save('spritemaps/spritemap_conv1_relu.jpeg', 'jpeg')
    map4.save('spritemaps/spritemap_conv1_relu_alpha.jpeg', 'jpeg')
    map5.save('spritemaps/spritemap_pool1.jpeg', 'jpeg')
    map6.save('spritemaps/spritemap_pool1_alpha.jpeg', 'jpeg')
    map7.save('spritemaps/spritemap_conv2.jpeg', 'jpeg')
    map8.save('spritemaps/spritemap_conv2_alpha.jpeg', 'jpeg')
    map9.save('spritemaps/spritemap_conv2_relu.jpeg', 'jpeg')
    map10.save('spritemaps/spritemap_conv2_relu_alpha.jpeg', 'jpeg')
    map11.save('spritemaps/spritemap_pool2.jpeg', 'jpeg')
    map12.save('spritemaps/spritemap_pool2_alpha.jpeg', 'jpeg')
    map13.save('spritemaps/spritemap_conv3.jpeg', 'jpeg')
    map14.save('spritemaps/spritemap_conv3_alpha.jpeg', 'jpeg')
    map15.save('spritemaps/spritemap_conv3_relu.jpeg', 'jpeg')
    map16.save('spritemaps/spritemap_conv3_relu_alpha.jpeg', 'jpeg')
    map17.save('spritemaps/spritemap_pool3.jpeg', 'jpeg')
    map18.save('spritemaps/spritemap_pool3_alpha.jpeg', 'jpeg')


if __name__ == "__main__":
    main()
