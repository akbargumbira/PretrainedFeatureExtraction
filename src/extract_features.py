# coding=utf-8
import numpy as np
from keras import applications
from keras.models import Model

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
    np.savez_compressed(path, bottleneck_features)


def generate_vgg16_cnn_codes(output_dir, prefix='training'):
    # Load dataset
    output_path = data_path(output_dir)
    t_data = load_dataset(output_path, prefix=prefix, data_only=True)

    # Preprocess training_data and get the bottleneck features
    net_model = applications.VGG16(weights='imagenet', include_top=False)
    t_data = applications.vgg16.preprocess_input(t_data)

    # Extract intermediate feature maps
    # layer_name_1 = 'block5_conv3'
    # layer_name_2 = 'block5_conv2'
    # layer_name_3 = 'block5_conv1'
    layer_name_4 = 'block4_pool'

    # Intermediate output 1
    # intermediate_layer_model_1 = Model(inputs=net_model.input,
    #                                  outputs=net_model.get_layer(layer_name_1).output)
    # intermediate_output_1 = intermediate_layer_model_1.predict(t_data)
    #
    # # Save the CNN codes
    # path = data_path(output_dir, 'cnn_vgg_last_1_%s.npz' % prefix)
    # np.savez_compressed(path, intermediate_output_1)
    # Intermediate output 2
    # intermediate_layer_model_2 = Model(inputs=net_model.input,
    #                                    outputs=net_model.get_layer(
    #                                        layer_name_2).output)
    # intermediate_output_2 = intermediate_layer_model_2.predict(t_data)
    # path = data_path(output_dir, 'cnn_vgg_last_2_%s.npz' % prefix)
    # np.savez_compressed(path, intermediate_output_2)
    #
    # # Intermediate output 3
    # intermediate_layer_model_3 = Model(inputs=net_model.input,
    #                                    outputs=net_model.get_layer(
    #                                        layer_name_3).output)
    # intermediate_output_3 = intermediate_layer_model_3.predict(t_data)
    # path = data_path(output_dir, 'cnn_vgg_last_3_%s.npz' % prefix)
    # np.savez_compressed(path, intermediate_output_3)

    # Intermediate output 4
    intermediate_layer_model_4 = Model(inputs=net_model.input,
                                       outputs=net_model.get_layer(
                                           layer_name_4).output)
    intermediate_output_4 = intermediate_layer_model_4.predict(t_data)
    path = data_path(output_dir, 'cnn_vgg_last_4_%s.npz' % prefix)
    np.savez_compressed(path, intermediate_output_4)
    print('test')


# Codalab Validation
generate_vgg16_cnn_codes(
    output_dir='codalab/224_224/',
    prefix='training'
)

generate_vgg16_cnn_codes(
    output_dir='codalab/224_224/',
    prefix='val'
)



# models = ['vgg16', 'inceptionv3', 'resnet50']
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
# # Kaggle Dog and Cat
# for model in models:
#     generate_bottleneck_features(
#         output_dir='kaggle_dog_cat/224_224',
#         model=model,
#         prefix='training'
#     )
