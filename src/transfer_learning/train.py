# coding=utf-8
import os
import time

import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint

from src.utilities import root_path, serialize_object


def preprocess_input(x):
    x = x.astype('float32')
    x /= 255.
    return x


def get_model(n_classes):
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3),
               padding='same',
               activation='relu',
               input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_classes, activation='softmax'))
    return model


def get_model2(n_classes):
    model = Sequential()
    model.add(
        Conv2D(16, (3, 3),
               padding='same',
               activation='relu',
               input_shape=(32, 32, 3)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))
    return model


def train(x_train, y_train, x_test, y_test, output_dir,
          batch_size=256, previous_model=None, epochs=200, initial_epoch=0):

    start_time = time.strftime("%Y%m%d-%H%M%S")
    abs_output_path = root_path(
        'src', 'transfer_learning', 'models', output_dir)
    os.makedirs(abs_output_path, exist_ok=True)

    # Preprocess data
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)
    # OHE the label
    n_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    # Prepare the model
    model = get_model2(n_classes)
    print(model.summary())
    # Use saved model weights if specified
    if previous_model:
        model.load_weights(previous_model)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Prepare callbacks
    # 1. ModelCheckpoint
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
    # 2. Tensorboard
    tensorboard_checkpoint = TensorBoard(
        log_dir=os.path.join(abs_output_path, 'tensorboard'))
    callbacks_list = [model_checkpoint, tensorboard_checkpoint]

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
    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train(x_train, y_train, x_test, y_test, 'test2', epochs=100)
