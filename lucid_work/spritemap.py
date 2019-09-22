from PIL import Image

def create_rows(layer, num_rows, num_cols, alpha=False):
    rows = []
    fn = 0
    for i in range(num_rows):
        row = Image.new('RGB', ((116*num_cols), (116)))
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
            offset += 116
        rows.append(row)
    return rows


def create_map(layer, num_rows, num_cols, alpha=False):
    map = Image.new('RGB', ((num_cols*116), (num_rows*116)))
    rows = create_rows(layer, num_rows, num_cols, alpha)
    offset = 0
    for i in range(num_rows):
        img = rows[i]
        map.paste(im=img, box=(0, offset))
        offset += 116
    return map

def main():
    # map1 = create_map('conv1', 10, 10, False)
    # map2 = create_map('conv1', 10, 10, True)
    # map1.save('spritemaps/spritemap_conv1.jpeg', 'jpeg')
    # map2.save('spritemaps/spritemap_conv1_alpha.jpeg', 'jpeg')
    map1 = create_map('conv1_relu', 10, 10, False)
    map2 = create_map('conv1_relu', 10, 10, True)
    map3 = create_map('pool1', 10, 10, False)
    map4 = create_map('pool1', 10, 10, True)
    map5 = create_map('conv2', 8, 12, False)
    map6 = create_map('conv2', 8, 12, True)
    map7 = create_map('conv2_relu', 8, 12, False)
    map8 = create_map('conv2_relu', 8, 12, True)
    map9 = create_map('pool2', 8, 12, False)
    map10 = create_map('pool2', 8, 12, True)
    map11 = create_map('conv3', 8, 12, False)
    map12 = create_map('conv3', 8, 12, True)
    map13 = create_map('conv3_relu', 8, 12, False)
    map14 = create_map('conv3_relu', 8, 12, True)
    map15 = create_map('pool3', 8, 12, False)
    map16 = create_map('pool3', 8, 12, True)
    map17 = create_map('fc1', 14, 25, False)
    map18 = create_map('fc1', 14, 25, False)
    map19 = create_map('fc1_relu', 14, 25, False)
    map20 = create_map('fc1_relu', 14, 25, False)

    map1.save('spritemaps/spritemap_conv1_relu.jpeg', 'jpeg')
    map2.save('spritemaps/spritemap_conv1_relu_alpha.jpeg', 'jpeg')
    map3.save('spritemaps/spritemap_pool1.jpeg', 'jpeg')
    map4.save('spritemaps/spritemap_pool1_alpha.jpeg', 'jpeg')
    map5.save('spritemaps/spritemap_conv2.jpeg', 'jpeg')
    map6.save('spritemaps/spritemap_conv2_alpha.jpeg', 'jpeg')
    map7.save('spritemaps/spritemap_conv2_relu.jpeg', 'jpeg')
    map8.save('spritemaps/spritemap_conv2_relu_alpha.jpeg', 'jpeg')
    map9.save('spritemaps/spritemap_pool2.jpeg', 'jpeg')
    map10.save('spritemaps/spritemap_pool2_alpha.jpeg', 'jpeg')
    map11.save('spritemaps/spritemap_conv3.jpeg', 'jpeg')
    map12.save('spritemaps/spritemap_conv3_alpha.jpeg', 'jpeg')
    map13.save('spritemaps/spritemap_conv3_relu.jpeg', 'jpeg')
    map14.save('spritemaps/spritemap_conv3_relu_alpha.jpeg', 'jpeg')
    map15.save('spritemaps/spritemap_pool3.jpeg', 'jpeg')
    map16.save('spritemaps/spritemap_pool3_alpha.jpeg', 'jpeg')
    map17.save('spritemaps/spritemap_fc1.jpeg', 'jpeg')
    map18.save('spritemaps/spritemap_fc1_alpha.jpeg', 'jpeg')
    map19.save('spritemaps/spritemap_fc1_relu.jpeg', 'jpeg')
    map20.save('spritemaps/spritemap_fc1_relu_alpha.jpeg', 'jpeg')

if __name__ == "__main__":
    main()
