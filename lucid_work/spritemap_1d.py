from PIL import Image

def create_rows(layer, num_rows, num_cols):
    rows = []
    fn = 0
    for i in range(num_rows):
        row = Image.new('RGBA', ((210*num_cols), (157)))
        offset = 0
        for i in range(num_cols):
            image_file = 'data/model_1d_vis/' + layer + '/resized/' + layer + '_' + str(fn) + '.png'
            fn += 1

            image = Image.open(image_file)
            row.paste(im=image, box=(offset, 0))
            offset += 210
        rows.append(row)
    return rows


def create_map(layer, num_rows, num_cols):
    map = Image.new('RGBA', ((num_cols*210), (num_rows*157)))
    rows = create_rows(layer, num_rows, num_cols)
    offset = 0
    for i in range(num_rows):
        img = rows[i]
        map.paste(im=img, box=(0, offset))
        offset += 157
    return map

def main():
    map1 = create_map("fc1", 35, 10)
    map2 = create_map("fc1_relu", 35, 10)
    map3 = create_map("fc2_1", 20, 10)
    map4 = create_map("fc2_1_relu", 20, 10)
    map5 = create_map("fc2_2", 35, 10)
    map6 = create_map("fc2_2_relu", 35, 10)
    map7 = create_map("fc2_3", 15, 10)
    map8 = create_map("fc2_3_relu", 15, 10)

    map1.save('data/model_1d_vis/1d_spritemaps/fc1.png')
    map2.save('data/model_1d_vis/1d_spritemaps/fc1_relu.png')
    map3.save('data/model_1d_vis/1d_spritemaps/fc2_1.png')
    map4.save('data/model_1d_vis/1d_spritemaps/fc2_1_relu.png')
    map5.save('data/model_1d_vis/1d_spritemaps/fc2_2.png')
    map6.save('data/model_1d_vis/1d_spritemaps/fc2_2_relu.png')
    map7.save('data/model_1d_vis/1d_spritemaps/fc2_3.png')
    map8.save('data/model_1d_vis/1d_spritemaps/fc2_3_relu.png')

    # map1 = create_map('conv1', 10, 10)
    # map2 = create_map('conv1_relu', 10, 10)
    # map3 = create_map('pool1', 10, 10)
    # map4 = create_map('conv2', 8, 12)
    # map5 = create_map('conv2_relu', 8, 12)
    # map6 = create_map('pool2', 8, 12)
    # map7 = create_map('conv3', 8, 12)
    # map8 = create_map('conv3_relu', 8, 12)
    # map9 = create_map('pool3', 8, 12)
    #
    #
    # map1.save('data/model_1d_vis/1d_spritemaps/conv1.png')
    # map2.save('data/model_1d_vis/1d_spritemaps/conv1_relu.png')
    # map3.save('data/model_1d_vis/1d_spritemaps/pool1.png')
    # map4.save('data/model_1d_vis/1d_spritemaps/conv2.png')
    # map5.save('data/model_1d_vis/1d_spritemaps/conv2_relu.png')
    # map6.save('data/model_1d_vis/1d_spritemaps/pool2.png')
    # map7.save('data/model_1d_vis/1d_spritemaps/conv3.png')
    # map8.save('data/model_1d_vis/1d_spritemaps/conv3_relu.png')


if __name__ == "__main__":
    main()
