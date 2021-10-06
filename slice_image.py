from PIL import Image
import numpy as np

def slice_image(im, height, width):
    image_width, image_height = im.size
    a = []
    for i in range(0, image_width, width):
        for j in range(0, image_height, height):
            box = (i, j, i+width, j+height)
            a.append(im.crop(box))

    return a

def concatenate_image(parts, height, width):
    im = Image.new('RGB', (width, height))
    stride_x, stride_y = parts[0].size
    idx = 0
    for i in range(0, width, stride_x):
        for j in range(0, height, stride_y):
            im.paste(parts[idx], (i,j))
            idx = idx+1

    return im

def to_array(images):
    l = len(images)
    width, height = images[0].size
    channels = 3
    x = np.ndarray((l, width, height, channels))
    for i in range(l):
        x[i] = np.asarray(images[i])
    
    x = x.astype('float32') / 255

    return x

def to_image(array):
    x = []
    for i in array:
        x.append(Image.fromarray((i * 255).astype(np.uint8)))
    return x
