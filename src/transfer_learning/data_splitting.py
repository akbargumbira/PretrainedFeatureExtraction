# coding=utf-8
import random
import os
from argparse import ArgumentParser

import numpy as np
from keras.datasets import cifar10

from src.utilities import root_path, serialize_object

CIFAR10_LABELS = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

TRANSPORT_LABELS = [0, 1, 8, 9]
ANIMAL_LABELS = [2, 3, 4, 5, 6, 7]


def get_subset_cifar(labels):
    """Get the subset of CIFAR given the labels.

    :param labels: List of labels.
    :type labels: list

    :return: Tuple of subset (x_train, y_train), (x_test, y_test)
    :rtype: tuple
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_subset_idx = np.where(np.isin(y_train.flatten(), labels))[0]
    test_subset_idx = np.where(np.isin(y_test.flatten(), labels))[0]

    return ((x_train[train_subset_idx], y_train[train_subset_idx]),
            (x_test[test_subset_idx, y_test[test_subset_idx]]))


def get_random_splits(seed):
    """Create random splits from CIFAR10.

    This random split should result in 2 group of labels that hopefully are
    similar. This is done by making sure that each of the group has 3 animal
    and 2 transport labels.

    :param seed: The seed for random splitting.
    :type seed: int
    """
    # Take 3 from animal, 2 from transport
    random.seed(seed)
    animals_chosen = random.sample(set(ANIMAL_LABELS), 3)
    transport_chosen = random.sample(set(TRANSPORT_LABELS), 2)
    groupA = sorted(animals_chosen + transport_chosen)
    groupB = sorted(list(set(range(10)) - set(groupA)))

    return groupA, groupB


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        '-r', help='Random split? If not specified, then transport-animal '
                   'splitting', action='store_true')
    parser.add_argument('-s', help='The seed for random splitting',
                        default=0, type=int)
    args = parser.parse_args()

    return args.r, args.s


if __name__ == "__main__":
    is_random, seed = parse_arguments()
    # is_random, seed = False, 0
    if is_random:
        print('Mode: Splitting labels `randomly`')
        groupA, groupB = get_random_splits(seed)
        all_labels = sorted(groupA + groupB)
        assert all_labels == list(range(10)), 'Missing some labels'

        prefix = 'half_rand'
        map_A_old_to_new = dict(zip(groupA, range(len(groupA))))
        map_A_file = root_path('src', 'transfer_learning', 'models', 'data',
                               '%s%s_A_labels_map.pkl' % (prefix, seed))
        serialize_object(map_A_old_to_new, map_A_file)

        map_B_old_to_new = dict(zip(groupB, range(len(groupB))))
        map_B_file = root_path('src', 'transfer_learning', 'models', 'data',
                               '%s%s_B_labels_map.pkl' % (prefix, seed))
        serialize_object(map_B_old_to_new, map_B_file)

    else:
        print('Mode: Splitting between animals and transport labels')
        prefix = 'half_anitrans'
        groupA, groupB = ANIMAL_LABELS, TRANSPORT_LABELS

        map_A_old_to_new = dict(zip(groupA, range(len(groupA))))
        map_A_file = root_path('src', 'transfer_learning', 'models', 'data',
                               '%s_A_labels_map.pkl' % prefix)
        serialize_object(map_A_old_to_new, map_A_file)

        map_B_old_to_new = dict(zip(groupB, range(len(groupB))))
        map_B_file = root_path('src', 'transfer_learning', 'models', 'data',
                               '%s_B_labels_map.pkl' % prefix)
        serialize_object(map_B_old_to_new, map_B_file)

