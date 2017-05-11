# coding=utf-8
import os
import fnmatch
import numpy as np
from keras.preprocessing import image as keras_image


def generate_dataset(image_dir):
    # Training data
    id, training_data, training_labels = [], [], []
    for root, dirnames, filenames in os.walk(image_dir):
        filenames = fnmatch.filter(filenames, '*.[Jj][Pp][Gg]')
        for index, filename in enumerate(filenames):
            image_path = os.path.join(root, filename)
            if os.path.exists(image_path):
                id.append(filename)
                image = keras_image.load_img(image_path, target_size=(224, 224))
                image_x = keras_image.img_to_array(image)
                image_x = np.expand_dims(image_x, axis=0)
                training_data.extend(image_x)
                if 'dog' in filename.lower():
                    training_labels.append(1)
                else:
                    training_labels.append(0)
    id_clean = np.char.replace(id, '.jpg', '')
    np.savez_compressed(open('model/training_id.npz', 'w'), np.array(id_clean))
    np.savez_compressed(open('model/training_data.npz', 'w'), np.array(
        training_data))
    np.savez_compressed(open('model/training_label.npz', 'w'), np.array(
        training_labels))

image_dir = '/home/agumbira/dev/data/dog_cat_kaggle/train'
generate_dataset(image_dir)
