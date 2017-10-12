# coding=utf-8
import os

import numpy as np
from keras import applications
from keras.applications.imagenet_utils import preprocess_input

from src.utilities import load_dataset, data_path


def generate_bottleneck_features(output_dir, model='vgg16', prefix='training'):
    # Load dataset
    output_path = data_path(output_dir)
    t_id, t_data, t_label = load_dataset(output_path, prefix=prefix)

    # Preprocess training_data
    t_data = preprocess_input(t_data)

    # Build the VGG16 network and get only the bottleneck features
    if model == 'vgg16':
        net_model = applications.VGG16(weights='imagenet', include_top=False)
    elif model == 'inceptionv3':
        net_model = applications.InceptionV3(weights='imagenet', include_top=False)
    elif model == 'resnet50':
        net_model = applications.ResNet50(weights='imagenet', include_top=False)
    else:
        raise ValueError('Model available: vgg16, inception_v3, resnet50')

    bottleneck_features = net_model.predict(t_data)

    # Save the bottleneck features
    path = data_path(output_dir, 'features_%s.npz' % model)
    np.savez_compressed(open(path, 'w'), bottleneck_features)


# vgg16
# generate_bottleneck_features(
#     output_dir='uiuc/224_224/',
#     model='vgg16',
#     prefix='training')

# inceptionv3
# generate_bottleneck_features(
#     output_dir='uiuc/224_224/',
#     model='inceptionv3',
#     prefix='training')

# # resnet50
# generate_bottleneck_features(
#     output_dir='uiuc/224_224/',
#     model='resnet50',
#     prefix='training')

# inceptionv3 299_299
generate_bottleneck_features(
    output_dir='uiuc/299_299/',
    model='inceptionv3',
    prefix='training')
