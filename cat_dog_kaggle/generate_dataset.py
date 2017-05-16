# coding=utf-8
import os
import fnmatch
import getpass
import numpy as np
from keras.preprocessing import image as keras_image


def generate_dataset(image_dir):
    # Training data
    id, training_labels = np.empty((0)), np.empty((0))
    training_data = []
    for root, dirnames, filenames in os.walk(image_dir):
        n_images = len(filenames)
        filenames = fnmatch.filter(filenames, '*.[Jj][Pp][Gg]')
        for index, filename in enumerate(filenames):
            n_chunk = int(round(float(n_images) / 100))
            n_chunk = 1 if n_chunk == 0 else n_chunk
            if index % n_chunk == 0:
                print 'Procesed: %s %% of images' % (index * 100 / n_images)
            image_path = os.path.join(root, filename)
            if os.path.exists(image_path):
                id = np.append(id, filename)
                image = keras_image.load_img(image_path, target_size=(150, 150))
                image = keras_image.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                training_data.extend(image)
                if 'dog' in filename.lower():
                    training_labels = np.append(training_labels, 1)
                else:
                    training_labels = np.append(training_labels, 0)
    print 'Saving the training id, data and label....'
    id = np.char.replace(id, '.jpg', '')
    np.savez_compressed(open('model/training_id.npz', 'w'), id)
    np.savez_compressed(
        open('model/training_data.npz', 'w'), np.array(training_data))
    np.savez_compressed(open('model/training_label.npz', 'w'), training_labels)

image_dir = '/home/%s/dev/data/dog_cat_kaggle/test_small' % getpass.getuser()
generate_dataset(image_dir)
