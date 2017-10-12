# coding=utf-8
import numpy as np
from keras import applications
from keras.applications.vgg16 import preprocess_input
from utilities import load_dataset

# Load dataset
training_id, training_data, training_label = load_dataset()

# Preprocess training_data
training_data = preprocess_input(training_data)

# Build the VGG16 network and get only the bottleneck features
model = applications.VGG16(weights='imagenet', include_top=False)
bottleneck_features = model.predict(training_data)

# Save the bottleneck features for training the top model later
np.savez_compressed(open('model/bottleneck_features.npz', 'w'), bottleneck_features)

