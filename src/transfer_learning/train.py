# coding=utf-8
import os
import time

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import cifar10

from src.utilities import root_path, serialize_object, load_serialized_object
from src.transfer_learning.data_splitting import get_subset_cifar, map_labels


def preprocess_input(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    return x_train, x_test


def get_model(n_classes, version=3):
    model = Sequential()

    if version == 1:
        model.add(
            Conv2D(32, (3, 3),
                   name='conv1',
                   padding='same',
                   activation='relu',
                   input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2), name='pool1'))
        model.add(Dropout(0.2))

        model.add(Conv2D(32, (3, 3),
                         name='conv2',
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling2D((2, 2), name='pool2'))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3),
                         name='conv3_1',
                         padding='same',
                         activation='relu'))
        model.add(Conv2D(128, (3, 3),
                         name='conv3_2',
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling2D((2, 2), name='pool3'))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(80, activation='relu', name='fc4'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu', name='fc5'))
        model.add(Dropout(0.2))
        model.add(Dense(n_classes, name='fc6', activation='softmax'))
    elif version == 2:
        model.add(
            Conv2D(32, (3, 3),
                   name='conv1_1',
                   padding='same',
                   activation='relu',
                   input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3),
                         name='conv1_2',
                         activation='relu'))
        model.add(MaxPooling2D((2, 2), name='pool1'))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3),
                         name='conv2_1',
                         activation='relu',
                         padding='same'))
        model.add(Conv2D(128, (3, 3),
                         name='conv2_2',
                         activation='relu'))
        model.add(MaxPooling2D((2, 2), name='pool2'))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3),
                         name='conv3',
                         activation='relu',
                         padding='same'))
        model.add(MaxPooling2D((2, 2), name='pool3'))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(64, activation='relu', name='fc4'))
        model.add(Dropout(0.2))
        model.add(Dense(n_classes, name='fc5', activation='softmax'))
    elif version == 3:
        model.add(
            Conv2D(32, (3, 3),
                   name='conv1_1',
                   padding='same',
                   activation='relu',
                   input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3),
                         name='conv1_2',
                         activation='relu'))
        model.add(MaxPooling2D((2, 2), name='pool1'))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3),
                         name='conv2_1',
                         activation='relu',
                         padding='same'))
        model.add(Conv2D(64, (3, 3),
                         name='conv2_2',
                         activation='relu'))
        model.add(MaxPooling2D((2, 2), name='pool2'))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3),
                         name='conv3_1',
                         activation='relu',
                         padding='same'))
        model.add(Conv2D(128, (3, 3),
                         name='conv3_2',
                         activation='relu'))
        model.add(MaxPooling2D((2, 2), name='pool3'))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(n_classes, name='fc4', activation='softmax'))
    elif version == 41:
        model.add(
            Conv2D(32, (3, 3),
                   name='conv1',
                   padding='same',
                   activation='relu',
                   input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2), name='pool1'))
        model.add(Dropout(0.2))

        model.add(Conv2D(32, (3, 3),
                         name='conv2',
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling2D((2, 2), name='pool2'))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3),
                         name='conv3_1',
                         padding='same',
                         activation='relu'))
        model.add(Conv2D(128, (3, 3),
                         name='conv3_2',
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling2D((4, 4), name='pool3'))
        model.add(Dropout(0.3))

        # Subject to compare
        model.add(Conv2D(128, (2, 2),
                         name='conv5',
                         padding='same',
                         activation='relu'))
        model.add(Conv2D(128, (2, 2),
                         name='conv6',
                         padding='same',
                         use_bias=False,
                         activation='relu'))
        model.add(Flatten())
        model.add(Dense(n_classes, name='fc_last', activation='softmax'))
    elif version == 42:
        model.add(
            Conv2D(32, (3, 3),
                   name='conv1',
                   padding='same',
                   activation='relu',
                   input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2), name='pool1'))
        model.add(Dropout(0.2))

        model.add(Conv2D(32, (3, 3),
                         name='conv2',
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling2D((2, 2), name='pool2'))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3),
                         name='conv3_1',
                         padding='same',
                         activation='relu'))
        model.add(Conv2D(128, (3, 3),
                         name='conv3_2',
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling2D((2, 2), name='pool3'))
        model.add(Dropout(0.3))

        # Subject to compare
        model.add(Conv2D(128, (2, 2),
                         name='conv5',
                         padding='same',
                         activation='relu'))
        model.add(Flatten())
        model.add(Dense(4*128, activation='relu', name='fc4', use_bias=False))
        model.add(Dense(n_classes, name='fc_last', activation='softmax'))
    elif version == 43:
        model.add(
            Conv2D(32, (3, 3),
                   name='conv1',
                   padding='same',
                   activation='relu',
                   input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2), name='pool1'))
        model.add(Dropout(0.2))

        model.add(Conv2D(32, (3, 3),
                         name='conv2',
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling2D((2, 2), name='pool2'))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3),
                         name='conv3_1',
                         padding='same',
                         activation='relu'))
        model.add(Conv2D(128, (3, 3),
                         name='conv3_2',
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling2D((2, 2), name='pool3'))
        model.add(Dropout(0.3))

    return model


def get_proper_position(weights_position, version=3):
    """Get the correct position of the layers given the weights position.

    ... weights_position = 1 -> 0
        weights_position = 2 -> 1
        weights_position = 3 -> 4
        weights_position = 4 -> 5
        weights_position = 5 -> 8
        weights_position = 6 -> 9

    :param weights_position: The position of the weights on the 'paper'.
    :type weights_position: int

    :return: The proper position based on the built model.
    :rtype: int
    """
    map_pos_1 = {1: 0, 2: 3, 3: 6, 4: 7, 5: 11, 6: 13}
    map_pos_2 = {1: 0, 2: 1, 3: 4, 4: 5, 5: 8, 6: 12}
    map_pos_3 = {1: 0, 2: 1, 3: 4, 4: 5, 5: 8, 6: 9}

    if not 1 <= weights_position < 7:
        raise ValueError('Weights position should be in [1, 6]')

    if version == 1:
        return map_pos_1[weights_position]
    elif version == 2:
        return map_pos_2[weights_position]
    elif version == 3:
        return map_pos_3[weights_position]


def get_prepared_model(n_classes, version=3, previous_model=None, is_base=False,
                       copied_pos=None, copied_weight_trainable=True):
    """Get prepared model.

    :param n_classes: The number of classes of the target task.
    :type n_classes: int

    :param previous_model: The path to previous model weights. In this case,
        the pretrained model on the base task.
    :type previous_model: str

    :param copied_pos: If specified, will copy the weights from 1 to
        copied_pos. The position is the same with on the 'paper'.
    :type copied_pos: int

    :return:
    """
    if is_base:
        # Training the netbase or half base
        model = get_model(n_classes, version)
    else:
        # Prepare the pretrained model of the other group
        model = get_model(10 - n_classes, version)

    if previous_model:
        model.load_weights(previous_model)

    if copied_pos:
        new_model = get_model(n_classes, version)
        layer_pos = get_proper_position(copied_pos, version)
        for i in list(range(layer_pos+1)):
            new_model.layers[i].set_weights(model.layers[i].get_weights())
            if not copied_weight_trainable:
                new_model.layers[i].trainable = False
        model = new_model

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def train(x_train, y_train, x_test, y_test, output_dir, version=3,
          is_base=False, batch_size=256, previous_model=None, copied_pos=None,
          copied_weight_trainable=True, checkpoint=False, tensorboard=False,
          epochs=200, initial_epoch=0):

    start_time = time.time()
    abs_output_path = root_path(
        'src', 'transfer_learning', 'models', output_dir)
    os.makedirs(abs_output_path, exist_ok=True)

    # Preprocess data
    x_train, x_test = preprocess_input(x_train, x_test)

    # OHE the label
    n_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    # Prepare the model
    model = get_prepared_model(
        n_classes, version, previous_model, is_base, copied_pos,
        copied_weight_trainable)

    # Prepare callbacks
    callbacks_list = []

    # 1. ModelCheckpoint
    if checkpoint:
        checkpoint_basedir = os.path.join(abs_output_path, 'checkpoint')
        os.makedirs(checkpoint_basedir, exist_ok=True)
        checkpoint_filepath = os.path.join(
            checkpoint_basedir,  'improved-{epoch:03d}-{val_acc:.2f}.hdf5')
        model_checkpoint = ModelCheckpoint(
            checkpoint_filepath,
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            mode='max')
        callbacks_list.append(model_checkpoint)

    # 2. Tensorboard
    if tensorboard:
        tensorboard_checkpoint = TensorBoard(
            log_dir=os.path.join(abs_output_path, 'tensorboard'))
        callbacks_list.append(tensorboard_checkpoint)

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
        verbose=2)

    # Dump the run time
    run_time = int(time.time() - start_time)
    serialize_object(run_time, os.path.join(abs_output_path, 'time.pkl'))
    # Dump the history
    serialize_object(
        history.history,
        os.path.join(abs_output_path, 'hist.pkl'))

    # Dump the last weights
    last_filepath = os.path.join(
        abs_output_path, 'weights-last.hdf5'.format(
            epoch=initial_epoch + epochs,
            val_acc=history.history['val_acc'][-1]))
    model.save_weights(last_filepath)


def get_prepared_data(labels_map_file):
    """Prepare the training and test data based on labels map file.

    :param labels_map_file: The path to the labels map.
    :type labels_map_file: str

    :return: The tuple of (x_train, y_train), (x_test, y_test)
    :rtype: tuple
    """
    labels_map = load_serialized_object(labels_map_file)
    old_labels = list(labels_map.keys())

    # Get all the data that have this old labels
    (x_train, y_train), (x_test, y_test) = get_subset_cifar(old_labels)

    # Change the labels using the map
    y_train = map_labels(y_train, labels_map)
    y_test = map_labels(y_test, labels_map)

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    # get_prepared_model(10, 1, is_base=True)
    epochs = 100
    model_versions = [1]
    for version in model_versions:
        # 1. Netbase
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        train(x_train, y_train, x_test, y_test, 'result_%s/netbase' % version,
              is_base=True, version=version, epochs=epochs)

        # 2-7. Half Random Base
        prefix = 'half_rand'
        groups = ['A', 'B']
        for seed in range(3):
            for group in groups:
                map_file = root_path('src', 'transfer_learning', 'models', 'data',
                                     '%s%s_%s_labels_map.pkl' % (prefix, seed, group))
                output_dir = 'result_%s/%s%s%s' % (version, prefix, seed, group)
                (x_train, y_train), (x_test, y_test) = get_prepared_data(map_file)
                train(x_train, y_train, x_test, y_test, output_dir,
                      is_base=True, version=version, epochs=epochs)

        # 8-9. Half Animal Transport Base
        prefix = 'half_anitrans'
        groups = ['A', 'B']
        for group in groups:
            map_file = root_path('src', 'transfer_learning', 'models', 'data',
                                 '%s_%s_labels_map.pkl' % (prefix, group))
            output_dir = 'result_%s/%s%s' % (version, prefix, group)
            (x_train, y_train), (x_test, y_test) = get_prepared_data(map_file)
            train(x_train, y_train, x_test, y_test, output_dir, epochs=epochs,
                  version=version, is_base=True)

        # TRANSFER
        # 1 + 2. SELFFER, not finetuned (36 models) + finetuned (36 models)
        prefix = 'half_rand'
        groups = ['A', 'B']
        finetuned_options = [False, True]
        for seed in range(3):
            for group in groups:
                map_file = root_path('src', 'transfer_learning', 'models', 'data',
                                     '%s%s_%s_labels_map.pkl' % (prefix, seed, group))
                (x_train, y_train), (x_test, y_test) = get_prepared_data(map_file)

                # Using pretrained model of the same group
                previous_model = root_path(
                    'src', 'transfer_learning', 'models', 'result_%s' % version,
                    '%s%s%s' % (prefix, seed, group),
                    'weights-last.hdf5')

                for layer in range(1, 7):
                    # Copy this layer weight only
                    for is_finetuned in finetuned_options:
                        if is_finetuned:
                            output_dir = 'result_%s/selffer_ft_%s%s%s%s_%s' % (
                                version, seed, group, seed, group, layer)
                        else:
                            output_dir = 'result_%s/selffer_%s%s%s%s_%s' % (
                                version, seed, group, seed, group, layer)

                        train(x_train, y_train, x_test, y_test,
                              output_dir,
                              version=version,
                              previous_model=previous_model,
                              copied_pos=layer,
                              copied_weight_trainable=is_finetuned,
                              epochs=epochs)

        # 3 + 4. TRANSFER A->B, B->A, not finetuned (36 models) + ft (36 models)
        prefix = 'half_rand'
        groups = ['A', 'B']
        finetuned_options = [False, True]
        for seed in range(3):
            for target_group in groups:
                # Target task: this group
                map_file = root_path('src', 'transfer_learning', 'models', 'data',
                                     '%s%s_%s_labels_map.pkl' % (
                                         prefix, seed, target_group))
                (x_train, y_train), (x_test, y_test) = get_prepared_data(map_file)

                # Using the pretrained model of: the other group
                base_group = list(set(groups) - set(list(target_group)))[0]
                previous_model = root_path(
                    'src', 'transfer_learning', 'models', 'result_%s' % version,
                    '%s%s%s' % (prefix, seed, base_group), 'weights-last.hdf5')

                for layer in range(1, 7):
                    for is_finetuned in finetuned_options:
                        if is_finetuned:
                            output_dir = 'result_%s/transfer_ft_%s%s%s%s_%s' % (
                                version, seed, base_group, seed, target_group, layer)
                        else:
                            output_dir = 'result_%s/transfer_%s%s%s%s_%s' % (
                                version, seed, base_group, seed, target_group, layer)

                        train(x_train, y_train, x_test, y_test,
                              output_dir,
                              version=version,
                              previous_model=previous_model,
                              copied_pos=layer,
                              copied_weight_trainable=is_finetuned,
                              epochs=epochs)

        # 5 + 6. TRANSFER Animal->Transport, Transport->Animal,
        # Not Finetuned + Finetuned (24 models)
        prefix = 'half_anitrans'
        groups = ['A', 'B']
        finetuned_options = [False, True]
        for target_group in groups:
            # Target task: this group
            map_file = root_path('src', 'transfer_learning', 'models',
                                 'data', '%s_%s_labels_map.pkl' % (
                                     prefix, target_group))
            (x_train, y_train), (x_test, y_test) = get_prepared_data(map_file)

            # Using the pretrained model of: the other group
            base_group = list(set(groups) - set(list(target_group)))[0]
            previous_model = root_path(
                'src', 'transfer_learning', 'models', 'result_%s' % version,
                '%s%s' % (prefix, base_group), 'weights-last.hdf5')

            for layer in range(1, 7):
                for is_finetuned in finetuned_options:
                    if is_finetuned:
                        output_dir = 'result_%s/transfer_ft_anitrans_%s%s_%s' % (
                            version, base_group, target_group, layer)
                    else:
                        output_dir = 'result_%s/transfer_anitrans_%s%s_%s' % (
                            version, base_group, target_group, layer)

                    train(x_train, y_train, x_test, y_test,
                          output_dir,
                          version=version,
                          previous_model=previous_model,
                          copied_pos=layer,
                          copied_weight_trainable=is_finetuned,
                          epochs=epochs)
