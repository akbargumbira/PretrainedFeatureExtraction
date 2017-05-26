# coding=utf-8
import numpy as np
from keras import applications
from keras.applications.vgg16 import preprocess_input, decode_predictions
from utilities import load_dataset


def generate_bottleneck_features(prefix='training'):
    # Load dataset
    t_id, t_data, t_gender_label, t_smile_label = load_dataset(prefix=prefix)

    # Preprocess training_data
    t_data = preprocess_input(t_data)

    # # Predictions
    # model_vgg16 = applications.VGG16(weights='imagenet')
    # prediction = model_vgg16.predict(training_data)
    # print('Predicted:', decode_predictions(prediction, top=3))

    # Build the VGG16 network and get only the bottleneck features
    model = applications.VGG16(weights='imagenet', include_top=False)
    bottleneck_features = model.predict(t_data)

    # Save the bottleneck features for training the top model later
    np.savez_compressed(
        open('model/%s_bottleneck_features.npz' % prefix, 'w'),
        bottleneck_features
    )

# Training
# generate_bottleneck_features()

# Validation
generate_bottleneck_features(prefix='val')
