# coding=utf-8
import os
import time

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint

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


def get_model(n_classes):
    model = Sequential()
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

    return model


def train(x_train, y_train, x_test, y_test, output_dir,
          batch_size=256, previous_model=None, checkpoint=False, epochs=200,
          initial_epoch=0):

    start_time = time.strftime("%Y%m%d-%H%M%S")
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
    model = get_model(n_classes)
    print(model.summary())
    # Use saved model weights if specified
    if previous_model:
        model.load_weights(previous_model)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

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

    # Dump the history
    end_time = time.strftime("%Y%m%d-%H%M%S")
    hist_file = 'hist_%s_%s.pkl' % (start_time, end_time)
    serialize_object(
        history.history,
        os.path.join(abs_output_path, hist_file))

    # Dump the last weights
    last_filepath = os.path.join(
        abs_output_path,
        'weights-last-{epoch:03d}-{val_acc:.2f}.hdf5'.format(
            epoch=initial_epoch + epochs,
            val_acc=history.history['val_acc'][-1]))
    model.save_weights(last_filepath)

if __name__ == '__main__':
    epochs = 100
    # 1. Netbase
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # train(x_train, y_train, x_test, y_test, 'result/netbase', epochs=epochs,
    #       checkpoint=True)

    # 2-7. Half Random Base
    # prefix = 'half_rand'
    # groups = ['A', 'B']
    # for seed in range(3):
    #     for group in groups:
    #         map_file = root_path('src', 'transfer_learning', 'models', 'data',
    #                              '%s%s_%s_labels_map.pkl' % (
    #                                    prefix, seed, group))
    #         labels_map = load_serialized_object(map_file)
    #         old_labels = list(labels_map.keys())
    #         # Get all the data that have this old labels
    #         (x_train, y_train), (x_test, y_test) = get_subset_cifar(old_labels)
    #         # Change the labels using the map
    #         y_train = map_labels(y_train, labels_map)
    #         y_test = map_labels(y_test, labels_map)
    #         output_dir = 'result/%s%s%s' % (prefix, seed, group)
    #         train(x_train, y_train, x_test, y_test, output_dir,
    #               epochs=epochs, checkpoint=True)

    # 8-9. Half Animal Transport Base
    # prefix = 'half_anitrans'
    # groups = ['A', 'B']
    # for group in groups:
    #     map_file = root_path('src', 'transfer_learning', 'models', 'data',
    #                          '%s_%s_labels_map.pkl' % (prefix, group))
    #     labels_map = load_serialized_object(map_file)
    #     old_labels = list(labels_map.keys())
    #     # Get all the data that have this old labels
    #     (x_train, y_train), (x_test, y_test) = get_subset_cifar(old_labels)
    #     # Change the labels using the map
    #     y_train = map_labels(y_train, labels_map)
    #     y_test = map_labels(y_test, labels_map)
    #     output_dir = 'result/%s%s' % (prefix, group)
    #     train(x_train, y_train, x_test, y_test, output_dir, epochs=epochs,
    #           checkpoint=True)

    # # TRANSFER
    # # 1. SELFFER, not finetuned
    # # prefix = 'half_rand'
    # groups = ['A', 'B']
    # for seed in range(3):
    #     for group in groups:
    #         map_file = root_path('src', 'transfer_learning', 'models', 'data',
    #                              '%s%s_%s_labels_map.pkl' % (
    #                                  prefix, seed, group))
    #         labels_map = load_serialized_object(map_file)
    #         old_labels = list(labels_map.keys())
    #         # Get all the data that have this old labels
    #         (x_train, y_train), (x_test, y_test) = get_subset_cifar(old_labels)
    #         # Change the labels using the map
    #         y_train = map_labels(y_train, labels_map)
    #         y_test = map_labels(y_test, labels_map)
    #
    #         for layer in range(6):
    #             # Copy this layer weight only
    #             previous_model =
    #
    #
    #             output_dir = 'result/%s%s%s' % (prefix, seed, group)
    #             train(x_train, y_train, x_test, y_test, output_dir,
    #                   epochs=epochs, checkpoint=True)

