# coding=utf-8
import os
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import RMSprop
from utilities import load_dataset


def train(previous_model=None, lr=0.001, epochs=50):
    """Train the CNN and save the model"""
    input_shape = (3, 50, 50) if K.image_data_format() == 'channels_first' else (50, 50, 3)

    # CNN network model
    model = Sequential()
    # Block 1
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation='relu',
               input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # Block 2
    model.add(
        Conv2D(64, kernel_size=(3, 3), activation='relu',
               input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # Flatten, 2 HL
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    if previous_model:
        model.load_weights(previous_model)

    model.compile(optimizer=RMSprop(lr=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Checkpoint
    filepath = 'model/crop/gender/weights-top_model-improvement-{epoch:02d}-{' \
               'val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # training data
    _, t_data, t_gender_label, _ = load_dataset(
        prefix='crop_training')

    # validation data
    _, v_data, v_gender_label, _ = load_dataset(
        prefix='crop_val')

    model.fit(
        t_data,
        t_gender_label,
        epochs=epochs,
        validation_data=(v_data, v_gender_label),
        callbacks=callbacks_list,
        verbose=True,
      )
    model.save_weights('model/crop/gender/weights-top_model_last.hdf5')

# First training
train(lr=0.01)
