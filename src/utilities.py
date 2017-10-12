# coding=utf-8
import os
import numpy as np
from keras.preprocessing import image as keras_image

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


def load_dataset(input_dir, prefix='training'):
    base_output = data_path(input_dir)
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
