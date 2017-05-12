# coding=utf-8
import os
import fnmatch
import getpass
import numpy as np
from keras.preprocessing import image as keras_image


def generate_dataset(image_dir):
    # Training data
    id, training_labels = np.empty((0)), np.empty((0))
    training_data = np.empty(shape=[0, 224, 224, 3])
    for root, dirnames, filenames in os.walk(image_dir):
        filenames = fnmatch.filter(filenames, '*.[Jj][Pp][Gg]')
        for index, filename in enumerate(filenames):
            image_path = os.path.join(root, filename)
            if os.path.exists(image_path):
                id = np.append(id, filename)
                image = keras_image.load_img(image_path, target_size=(224, 224))
                image = keras_image.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                training_data = np.vstack((training_data, image))
                if 'dog' in filename.lower():
                    training_labels = np.append(training_labels, 1)
                else:
                    training_labels = np.append(training_labels, 0)
    id = np.char.replace(id, '.jpg', '')
    np.savez_compressed(open('model/training_id.npz', 'w'), id)
    np.savez_compressed(open('model/training_data.npz', 'w'), training_data)
    np.savez_compressed(open('model/training_label.npz', 'w'), training_labels)

image_dir = '/home/%s/dev/data/dog_cat_kaggle/train' % getpass.getuser()
generate_dataset(image_dir)
