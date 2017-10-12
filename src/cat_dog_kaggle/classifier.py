# coding=utf-8
import numpy as np
from keras.models import Sequential
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

TOP_MODEL_WEIGHTS_PATH = 'model/weights-top_model-improvement-47-0.97.hdf5'
FINAL_MODEL_WEIGHTS_PATH = ''


class DogCatClassifier(object):
    """Class DogCat"""
    def __init__(self):
        """The constructor."""
        # Construct the model
        input_tensor = Input(shape=(150, 150, 3))
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor)

        # Add the top model layer that we have already trained
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1, activation='sigmoid'))
        top_model.load_weights(TOP_MODEL_WEIGHTS_PATH)

        # add the model on top of the convolutional base
        self._model = Model(
            input=base_model.input,
            output=top_model(base_model.output))

        # self._model.load_weights(FINAL_MODEL_WEIGHTS_PATH)

    def predict(self, data):
        predictions = self._model.predict(data)
        return predictions.ravel()
