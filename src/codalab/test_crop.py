# coding=utf-8
import os
import numpy as np
from keras.preprocessing import image as keras_image

image_dir = '/home/akbar/dev/data/codalab/test_small'
reference_file = 'gender_fex_valset.csv'
reference_path = os.path.join(image_dir, reference_file)
data = np.genfromtxt(
    reference_path,
    delimiter=',',
    skip_header=1,
    dtype='|S32, int, int, int, int, int, int'
)

# Data
for i in range(len(data)):
    x, y = data[i][1], data[i][2]
    width, height = data[i][3], data[i][4]

    image_filename = data[i][0]
    image_path = os.path.join(image_dir, image_filename)
    if os.path.exists(image_path):
        img = keras_image.load_img(image_path)
        img = img.crop((x, y, x+width, y+height))
        img.save(os.path.join(image_dir, 'crop', image_filename), 'jpeg')

