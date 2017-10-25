# coding=utf-8
import os
import fnmatch
import pickle

import numpy as np
from keras.preprocessing import image as keras_image

DATASETS = {
    1: 'uiuc',
    2: 'codalab - gender',
    3: 'codalab - smile',
    4: 'kaggle - dog cat'

}

UIUC_EVENT_CLASS = {
    'badminton': 0,
    'bocce': 1,
    'croquet': 2,
    'polo': 3,
    'rowing': 4,
    'RockClimbing': 5,
    'sailing': 6,
    'snowboarding': 7
}


def get_uiuc_training_data(image_dir, target_size):
    """Get UIUC training data and its label.

    The structure of UIUC images in the directory:
    __ event_img
    ___ badminton
    ___ bocce
    ___ etc.
    """
    training_id, training_data, training_labels = [], [], []
    for group_dir in UIUC_EVENT_CLASS.keys():
        path = os.path.join(os.curdir, image_dir, group_dir)
        files = os.listdir(path)
        for filename in files:
            if filename.lower().endswith('.jpg'):
                image_path = os.path.join(path, filename)
                if os.path.exists(image_path):
                    training_id = np.append(training_id, filename)
                    image = keras_image.load_img(
                        image_path,
                        target_size=target_size)
                    image = keras_image.img_to_array(image)
                    image = np.expand_dims(image, axis=0)
                    training_data.extend(image)
                    training_labels.append(UIUC_EVENT_CLASS[group_dir])

    return training_id, training_data, training_labels


def generate_codalab_dataset(image_dir, target_size, output_dir,
                             reference_file, crop=False, prefix='training'):
    reference_path = os.path.join(image_dir, reference_file)
    data = np.genfromtxt(
        reference_path,
        delimiter=',',
        skip_header=1,
        dtype='|S32, int, int, int, int, int, int'
    )

    # Data
    id, gender_labels, smile_labels = [], [], []
    training_data = []
    for i in range(len(data)):
        # Print the progress
        n_chunk = int(round(float(len(data)) / 100))
        n_chunk = 1 if n_chunk == 0 else n_chunk
        if i % n_chunk == 0:
            print 'Procesed: %s %% of images' % (i * 100 / len(data))

        # Get the data
        image_filename = data[i][0]
        image_path = os.path.join(image_dir, image_filename)
        if os.path.exists(image_path):
            # Labels and id
            gender_label = data[i][5]
            smile_label = data[i][6]
            id = np.append(id, image_filename)
            gender_labels = np.append(gender_labels, gender_label)
            smile_labels = np.append(smile_labels, smile_label)
            # Image
            if crop:
                x, y = data[i][1], data[i][2]
                width, height = data[i][3], data[i][4]
                # Don't resize but crop first
                image = keras_image.load_img(image_path)
                image = image.crop((x, y, x + width, y + height))
                # Now resize
                hw_tuple = (target_size[1], target_size[0])
                if image.size != hw_tuple:
                    image = image.resize(hw_tuple)
            else:
                image = keras_image.load_img(image_path,
                                             target_size=target_size)
            image = keras_image.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            training_data.extend(image)

    print 'Saving the training id, data and label....'
    base_output = data_path(output_dir)
    np.savez_compressed(
        open(os.path.join(base_output, '%s_id.npz' % prefix), 'w'),
        id)
    np.savez_compressed(
        open(os.path.join(base_output, '%s_data.npz' % prefix), 'w'),
        training_data)
    np.savez_compressed(
        open(os.path.join(base_output, '%s_gender_label.npz' % prefix), 'w'),
        gender_labels)
    np.savez_compressed(
        open(os.path.join(base_output, '%s_smile_label.npz' % prefix), 'w'),
        smile_labels)


def get_kaggle_dog_cat_data(image_dir, target_size):
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
                image = keras_image.load_img(
                    image_path, target_size=target_size)
                image = keras_image.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                training_data.extend(image)
                if 'dog' in filename.lower():
                    training_labels = np.append(training_labels, 1)
                else:
                    training_labels = np.append(training_labels, 0)
    return id, training_data, training_labels


def save_dataset(data, output_dir, prefix='training'):
    base_output = data_path(output_dir)
    np.savez_compressed(
        open(os.path.join(base_output, '%s_id.npz' % prefix), 'w'),
        data[0])
    np.savez_compressed(
        open(os.path.join(base_output, '%s_data.npz' % prefix), 'w'),
        data[1])
    np.savez_compressed(
        open(os.path.join(base_output, '%s_label.npz' % prefix), 'w'),
        data[2])


def load_dataset(input_dir, prefix='training', data_only=False):
    base_output = data_path(input_dir)

    if data_only:
        with np.load(os.path.join(base_output, '%s_data.npz' % prefix)) as data:
            training_data = data['arr_0.npy']
            return training_data
    else:
        with np.load(os.path.join(base_output, '%s_id.npz' % prefix)) as data:
            training_id = data['arr_0.npy']

        with np.load(os.path.join(base_output, '%s_data.npz' % prefix)) as data:
            training_data = data['arr_0.npy']

        with np.load(os.path.join(base_output, '%s_label.npz' % prefix)) as data:
            training_label = data['arr_0.npy']

        return training_id, training_data, training_label


def root_path(*args):
    """Get the path to the project root.

    :param args : List of path elements e.g. ['data', 'data.csv']

    :return: Absolute path to the project root.
    """
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir))
    for item in args:
        path = os.path.abspath(os.path.join(path, item))

    return path


def data_path(*args):
    """Get the path to a data file.

    :param args : List of path elements e.g. ['uiuc', 'model']

    :return: Absolute path to a specific data path.
    """
    path = root_path('src', 'data')
    for item in args:
        path = os.path.abspath(os.path.join(path, item))

    return path


def serialize_object(obj, output_path):
    """Serialize object into the specified output file."""
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def load_serialized_object(input_path):
    """Load serialized object from the specified path"""
    with open(input_path, "rb") as f:
        obj = pickle.load(f)
    return obj
