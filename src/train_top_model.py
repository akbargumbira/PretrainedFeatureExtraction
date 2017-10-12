# coding=utf-8
import os
import time

import numpy as np
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from src.utilities import data_path, serialize_object


def get_model(input_shape, n_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
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

    # Split into train and validation set
    train_data, val_data, train_label, val_label = train_test_split(
        training_data, training_label, random_state=42, stratify=training_label)

    model = get_model(training_data.shape[1:], n_classes)
    model.compile(optimizer='rmsprop',
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

    history = model.fit(
        train_data,
        train_label,
        epochs=50,
        validation_data=(val_data, val_label),
        callbacks=callbacks_list,
        verbose=True)

    # Dump the history
    hist_file = 'hist-%s.pkl' % time.strftime("%Y%m%d-%H%M%S")
    serialize_object(
        history.history,
        os.path.join(abs_output_path, hist_file))
    # Dump the last weights
    last_filepath = os.path.join(
        abs_output_path,
        'weights-top_model_last.hdf5')
    model.save_weights(last_filepath)


# VGG16
# train(
#     features_path='uiuc/224_224/features_vgg16.npz',
#     label_path='uiuc/224_224/training_label.npz',
#     output_dir='uiuc/224_224/model/vgg16/oct12')

# inceptionv3
# train(
#     features_path='uiuc/224_224/features_inceptionv3.npz',
#     label_path='uiuc/224_224/training_label.npz',
#     output_dir='uiuc/224_224/model/inceptionv3/oct12')

# resnet50
# train(
#     features_path='uiuc/224_224/features_resnet50.npz',
#     label_path='uiuc/224_224/training_label.npz',
#     output_dir='uiuc/224_224/model/resnet50/oct12')

# inceptionv3 299_299
train(
    features_path='uiuc/299_299/features_inceptionv3.npz',
    label_path='uiuc/299_299/training_label.npz',
    output_dir='uiuc/299_299/model/inceptionv3/oct12')
