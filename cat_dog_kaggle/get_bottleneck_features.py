# coding=utf-8
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras import applications
from utilities import load_dataset

training_id, training_data, training_label = load_dataset()
# Build the VGG16 network and get only the bottleneck features
model = applications.VGG16(weights='imagenet', include_top=False)
bottleneck_features = model.predict(training_data)

# Save the bottleneck features for training the top model later
np.savez_compressed(open('model/bottleneck_features.npz', 'w'), bottleneck_features)

