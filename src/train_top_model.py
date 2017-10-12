# coding=utf-8
import os

import numpy as np
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras.utils import to_categorical

from src.utilities import data_path


def get_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train(features_path, label_path, output_dir):
    abs_features_path = data_path(features_path)
    abs_label_path = data_path(label_path)
    abs_output_path = data_path(output_dir)

    # Get the bottleneck features
    with np.load(abs_features_path) as data:
        training_data = data['arr_0.npy']

    # Get the label
    with np.load(abs_label_path) as data:
        training_label = data['arr_0.npy']

    # OHE training label
    n_classes = len(np.unique(training_label))
    training_label = to_categorical(training_label, n_classes)
    model = get_model(training_data.shape[1:])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Checkpoint
    checkpoint_filepath = os.path.join(
        abs_output_path,
        'weights-top_model-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
    checkpoint = ModelCheckpoint(
        checkpoint_filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    callbacks_list = [checkpoint]

    model.fit(
        training_data, training_label,
        epochs=50,
        validation_split=0.3,
        callbacks=callbacks_list,
        verbose=True)

    last_filepath = os.path.join(
        abs_output_path,
        'weights-top_model_last.hdf5'

    )
    model.save_weights(last_filepath)


train(
    features_path='uiuc/224_224/training_bottleneck_features.npz',
    label_path='uiuc/224_224/training_label.npz',
    output_dir='uiuc/224_224/model')
