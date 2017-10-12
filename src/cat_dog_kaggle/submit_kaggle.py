# coding=utf-8
import os
import fnmatch
import getpass
import numpy as np
from keras.preprocessing import image as keras_image
import keras.backend as K
from keras.applications.vgg16 import preprocess_input
from classifier import DogCatClassifier

test_dir = '/home/agumbira/dev/data/dog_cat_kaggle/test'


def get_test_data(image_dir):
    # Training data
    id = np.empty((0))
    dataset = []
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
                dataset.extend(image)

    return id,  np.array(dataset)

id, dataset = get_test_data(test_dir)
dataset = preprocess_input(dataset)
classifier = DogCatClassifier()
predictions = classifier.predict(dataset)

id_clean = np.char.replace(id, '.jpg', '').astype(int)

data = np.column_stack((id_clean, predictions))
data = data[data[:, 0].argsort()]
np.savetxt('model/submission.csv', data, delimiter=',', fmt='%i, %1.1f',
           header='id,label', comments='')
