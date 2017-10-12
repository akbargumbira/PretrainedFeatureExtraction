# coding=utf-8
import numpy as np


def load_dataset(prefix='training'):
    with np.load('model/%s_id.npz' % prefix) as data:
        id = data['arr_0.npy']

    with np.load('model/%s_data.npz' % prefix) as data:
        t_data = data['arr_0.npy']

    with np.load('model/%s_gender_label.npz' % prefix) as data:
        gender_label = data['arr_0.npy']

    with np.load('model/%s_smile_label.npz' % prefix) as data:
        smile_label = data['arr_0.npy']

    return id, t_data, gender_label, smile_label
