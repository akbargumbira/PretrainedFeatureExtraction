# coding=utf-8
import os

import numpy as np
from keras import applications
from keras.applications.vgg16 import preprocess_input, decode_predictions

from src.utilities import load_dataset, data_path


def generate_vgg16_bottleneck_features(output_dir, prefix='training'):
    # Load dataset
    t_id, t_data, t_label = load_dataset('uiuc/224_224', prefix=prefix)

    # Preprocess training_data
    t_data = preprocess_input(t_data)

    # # Predictions
    # model_vgg16 = applications.VGG16(weights='imagenet')
    # prediction = model_vgg16.predict(t_data)
    # print('Predicted:', decode_predictions(prediction, top=3))

    # Build the VGG16 network and get only the bottleneck features
    model = applications.VGG16(weights='imagenet', include_top=False)
    bottleneck_features = model.predict(t_data)

    # Save the bottleneck features
    path = data_path(output_dir, '%s_bottleneck_features.npz' % prefix)
    np.savez_compressed(open(path, 'w'), bottleneck_features)


# Training
generate_vgg16_bottleneck_features(output_dir='uiuc/224_224',
                                   prefix='training')

# Validation
# generate_bottleneck_features(prefix='val')
