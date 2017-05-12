# coding=utf-8
import numpy as np


def load_dataset():
    with np.load('model/training_id.npz') as data:
        training_id = data['arr_0.npy']

    with np.load('model/training_data.npz') as data:
        training_data = data['arr_0.npy']

    with np.load('model/training_label.npz') as data:
        training_label = data['arr_0.npy']

    return training_id, training_data, training_label
