# coding=utf-8
import numpy as np
from keras import applications

from src.utilities import load_dataset, data_path


def generate_bottleneck_features(output_dir, model='vgg16', prefix='training'):
    # Load dataset
    output_path = data_path(output_dir)
    t_data = load_dataset(output_path, prefix=prefix, data_only=True)

    # Build the VGG16 network and get only the bottleneck features
    if model == 'vgg16':
        net_model = applications.VGG16(weights='imagenet', include_top=False)
        preprocess_input = applications.vgg16.preprocess_input
    elif model == 'inceptionv3':
        net_model = applications.InceptionV3(weights='imagenet', include_top=False)
        preprocess_input = applications.inception_v3.preprocess_input
    elif model == 'resnet50':
        net_model = applications.ResNet50(weights='imagenet', include_top=False)
        preprocess_input = applications.resnet50.preprocess_input
    else:
        raise ValueError('Model available: vgg16, inception_v3, resnet50')

    # Preprocess training_data and get the bottleneck features
    t_data = preprocess_input(t_data)
    bottleneck_features = net_model.predict(t_data)

    # Save the bottleneck features
    path = data_path(output_dir, 'features_%s_%s.npz' % (prefix, model))
    np.savez_compressed(open(path, 'w'), bottleneck_features)


models = ['vgg16', 'inceptionv3', 'resnet50']
# # UIUC
# for model in models:
#     generate_bottleneck_features(
#         output_dir='uiuc/224_224/',
#         model=model,
#         prefix='training'
#     )
# # ----------------------------
# # Codalab Training
# for model in models:
#     generate_bottleneck_features(
#         output_dir='codalab/224_224/',
#         model=model,
#         prefix='training'
#     )
#
# # Codalab Validation
# for model in models:
#     generate_bottleneck_features(
#         output_dir='codalab/224_224/',
#         model=model,
#         prefix='val'
#     )
# -------------------------------------
# Kaggle Dog and Cat
for model in models:
    generate_bottleneck_features(
        output_dir='kaggle_dog_cat/224_224',
        model=model,
        prefix='training'
    )
