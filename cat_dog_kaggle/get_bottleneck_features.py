import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras import applications


def load_dataset():
    with np.load('model/training_id.npz') as data:
        training_id = data['arr_0.npy']

    with np.load('model/training_data.npz') as data:
        training_data = data['arr_0.npy']

    with np.load('model/training_label.npz') as data:
        training_label = data['arr_0.npy']

    return training_id, training_data, training_label

training_id, training_data, training_label = load_dataset()
# build the VGG16 network
model = applications.VGG16(weights='imagenet')
predictions = model.predict(training_data)
print training_id
print training_label
print('Predicted:', decode_predictions(predictions, top=3))
