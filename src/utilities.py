# coding=utf-8
import os
import fnmatch
import pickle

import h5py
import numpy as np
from keras import applications
from keras.models import Sequential, Model
from keras.preprocessing import image as keras_image
from keras.layers import Dropout, Flatten, Dense, Input

DATASETS = {
    1: 'uiuc',
    2: 'codalab - gender',
    3: 'codalab - smile',
    4: 'kaggle - dog cat'

}

UIUC_EVENT_CLASS = {
    'badminton': 0,
    'bocce': 1,
    'croquet': 2,
    'polo': 3,
    'rowing': 4,
    'RockClimbing': 5,
    'sailing': 6,
    'snowboarding': 7
}

CODALAB_GENDER_CLASS = {
    0: 'Male',
    1: 'Female',
    2: 'Unknown'
}

CODALAB_SMILE_CLASS = {
    0: 'Not smiling',
    1: 'Smiling'
}

KAGGLE_DOGCAT_CLASS = {
    0: 'Cat',
    1: 'Dog'
}

def get_uiuc_training_data(image_dir, target_size):
    """Get UIUC training data and its label.

    The structure of UIUC images in the directory:
    __ event_img
    ___ badminton
    ___ bocce
    ___ etc.
    """
    training_id, training_data, training_labels = [], [], []
    for group_dir in UIUC_EVENT_CLASS.keys():
        path = os.path.join(os.curdir, image_dir, group_dir)
        files = os.listdir(path)
        for filename in files:
            if filename.lower().endswith('.jpg'):
                image_path = os.path.join(path, filename)
                if os.path.exists(image_path):
                    training_id = np.append(training_id, filename)
                    image = keras_image.load_img(
                        image_path,
                        target_size=target_size)
                    image = keras_image.img_to_array(image)
                    image = np.expand_dims(image, axis=0)
                    training_data.extend(image)
                    training_labels.append(UIUC_EVENT_CLASS[group_dir])

    return training_id, training_data, training_labels


def generate_codalab_dataset(image_dir, target_size, output_dir,
                             reference_file, crop=False, prefix='training'):
    reference_path = os.path.join(image_dir, reference_file)
    data = np.genfromtxt(
        reference_path,
        delimiter=',',
        skip_header=1,
        dtype='|S32, int, int, int, int, int, int'
    )

    # Data
    id, gender_labels, smile_labels = [], [], []
    training_data = []
    for i in range(len(data)):
        # Print the progress
        n_chunk = int(round(float(len(data)) / 100))
        n_chunk = 1 if n_chunk == 0 else n_chunk
        if i % n_chunk == 0:
            print('Procesed: %s %% of images' % (i * 100 / len(data)))

        # Get the data
        image_filename = data[i][0]
        image_path = os.path.join(image_dir, image_filename)
        if os.path.exists(image_path):
            # Labels and id
            gender_label = data[i][5]
            smile_label = data[i][6]
            id = np.append(id, image_filename)
            gender_labels = np.append(gender_labels, gender_label)
            smile_labels = np.append(smile_labels, smile_label)
            # Image
            if crop:
                x, y = data[i][1], data[i][2]
                width, height = data[i][3], data[i][4]
                # Don't resize but crop first
                image = keras_image.load_img(image_path)
                image = image.crop((x, y, x + width, y + height))
                # Now resize
                hw_tuple = (target_size[1], target_size[0])
                if image.size != hw_tuple:
                    image = image.resize(hw_tuple)
            else:
                image = keras_image.load_img(image_path,
                                             target_size=target_size)
            image = keras_image.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            training_data.extend(image)

    print('Saving the training id, data and label....')
    base_output = data_path(output_dir)
    np.savez_compressed(
        open(os.path.join(base_output, '%s_id.npz' % prefix), 'w'),
        id)
    np.savez_compressed(
        open(os.path.join(base_output, '%s_data.npz' % prefix), 'w'),
        training_data)
    np.savez_compressed(
        open(os.path.join(base_output, '%s_gender_label.npz' % prefix), 'w'),
        gender_labels)
    np.savez_compressed(
        open(os.path.join(base_output, '%s_smile_label.npz' % prefix), 'w'),
        smile_labels)


def get_kaggle_dog_cat_data(image_dir, target_size):
    # Training data
    id, training_labels = np.empty((0)), np.empty((0))
    training_data = []
    for root, dirnames, filenames in os.walk(image_dir):
        n_images = len(filenames)
        filenames = fnmatch.filter(filenames, '*.[Jj][Pp][Gg]')
        for index, filename in enumerate(filenames):
            n_chunk = int(round(float(n_images) / 100))
            n_chunk = 1 if n_chunk == 0 else n_chunk
            if index % n_chunk == 0:
                print('Procesed: %s %% of images' % (index * 100 / n_images))
            image_path = os.path.join(root, filename)
            if os.path.exists(image_path):
                id = np.append(id, filename)
                image = keras_image.load_img(
                    image_path, target_size=target_size)
                image = keras_image.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                training_data.extend(image)
                if 'dog' in filename.lower():
                    training_labels = np.append(training_labels, 1)
                else:
                    training_labels = np.append(training_labels, 0)
    return id, training_data, training_labels


def save_dataset(data, output_dir, prefix='training'):
    base_output = data_path(output_dir)
    np.savez_compressed(
        open(os.path.join(base_output, '%s_id.npz' % prefix), 'w'),
        data[0])
    np.savez_compressed(
        open(os.path.join(base_output, '%s_data.npz' % prefix), 'w'),
        data[1])
    np.savez_compressed(
        open(os.path.join(base_output, '%s_label.npz' % prefix), 'w'),
        data[2])


def load_dataset_from_path(rel_path):
    abs_path = data_path(rel_path)

    with np.load(abs_path) as data:
        t_data = data['arr_0.npy']
        return t_data


def load_dataset(input_dir, prefix='training', data_only=False):
    base_output = data_path(input_dir)

    if data_only:
        with np.load(os.path.join(base_output, '%s_data.npz' % prefix)) as data:
            training_data = data['arr_0.npy']
            return training_data
    else:
        with np.load(os.path.join(base_output, '%s_id.npz' % prefix)) as data:
            training_id = data['arr_0.npy']

        with np.load(os.path.join(base_output, '%s_data.npz' % prefix)) as data:
            training_data = data['arr_0.npy']

        with np.load(os.path.join(base_output, '%s_label.npz' % prefix)) as data:
            training_label = data['arr_0.npy']

        return training_id, training_data, training_label


def root_path(*args):
    """Get the path to the project root.

    :param args : List of path elements e.g. ['data', 'data.csv']

    :return: Absolute path to the project root.
    """
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir))
    for item in args:
        path = os.path.abspath(os.path.join(path, item))

    return path


def data_path(*args):
    """Get the path to a data file.

    :param args : List of path elements e.g. ['uiuc', 'model']

    :return: Absolute path to a specific data path.
    """
    path = root_path('src', 'data')
    for item in args:
        path = os.path.abspath(os.path.join(path, item))

    return path


def serialize_object(obj, output_path):
    """Serialize object into the specified output file."""
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def load_serialized_object(input_path):
    """Load serialized object from the specified path"""
    with open(input_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def plot_model():
    from keras.utils import plot_model
    from keras.applications.vgg16 import VGG16
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.resnet50 import ResNet50

    model = VGG16(weights='imagenet', include_top=False)
    plot_model(model, to_file='vgg16.png', show_shapes=True)

    model = InceptionV3(weights='imagenet', include_top=False)
    plot_model(model, to_file='inceptionv3.png', show_shapes=True)

    model = ResNet50(weights='imagenet', include_top=False)
    plot_model(model, to_file='resnet50.png', show_shapes=True)


def print_model_summary(training_features_path, training_label_path,
          val_features_path=None, val_label_path=None):
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    training_data, training_label = load_bottleneck_features(
        training_features_path, training_label_path)

    n_classes = len(np.unique(training_label))
    assert n_classes >= 2, 'n_classes should be >= 2, got %s' % n_classes
    if n_classes > 2:
        # OHE training label
        training_label = to_categorical(training_label, n_classes)

    if val_features_path:
        val_data, val_label = load_bottleneck_features(
            val_features_path, val_label_path)
        train_data, train_label = training_data, training_label
        if n_classes > 2:
            val_label = to_categorical(val_label, n_classes)
    else:
        # Split into train and validation set
        train_data, val_data, train_label, val_label = train_test_split(
            training_data, training_label, random_state=42,
            stratify=training_label)

    # Prepare the model
    model = get_top_model(train_data.shape[1:], n_classes)
    model.summary()


def get_full_model(image_size, base_architecture, n_target_classes,
                   top_model_weights_path):
    abs_top_model_weights_path = data_path(top_model_weights_path)
    input_size = (image_size[0], image_size[1], 3)
    input_tensor = Input(shape=input_size)

    # Build the VGG16 network and get only the bottleneck features
    if base_architecture == 'vgg16':
        base_model = applications.VGG16(
            weights='imagenet', include_top=False, input_tensor=input_tensor)
    elif base_architecture == 'inceptionv3':
        base_model = applications.InceptionV3(
            weights='imagenet', include_top=False, input_tensor=input_tensor)
    elif base_architecture == 'resnet50':
        base_model = applications.ResNet50(
            weights='imagenet', include_top=False, input_tensor=input_tensor)
    else:
        raise ValueError('Model available: vgg16, inception_v3, resnet50')

    # The top model
    top_model = get_top_model(
        input_shape=base_model.output_shape[1:],  n_classes=n_target_classes)
    top_model.load_weights(abs_top_model_weights_path)

    # Add the model on top of the convolutional base
    model = Model(
        input=base_model.input,
        output=top_model(base_model.output))

    return model


def get_top_model(input_shape, n_classes):
    """Create a simple fully connected layers (2HL) to train extracted
        ConvNet features.

    :param input_shape: The shape of the input (the output of ConvNet layer).
    :type input_shape: int

    :param n_classes: The number of classes in target classification.
    :type n_classes: int

    :return:
    """
    assert n_classes >= 2, 'n_classes should be >= 2, got %s' % n_classes
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    if n_classes > 2:
        model.add(Dense(n_classes, activation='softmax'))
    else:
        model.add(Dense(1, activation='sigmoid'))
    return model


def load_bottleneck_features(features_path, label_path):
    abs_features_path = data_path(features_path)
    abs_label_path = data_path(label_path)

    # Get the bottleneck features and its labels
    with np.load(abs_features_path) as data:
        features = data['arr_0.npy']
    with np.load(abs_label_path) as data:
        label = data['arr_0.npy']

    return features, label


def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in iter(f.attrs.items()):
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return

        for layer, g in iter(f.items()):
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in iter(g.keys()):
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()


# # plot_model()
# models = ['vgg16', 'inceptionv3', 'resnet50']
# for model in models:
#     print model
#     print_model_summary(
#         training_features_path='uiuc/224_224/features_training_%s.npz' %
#                                model,
#         training_label_path='uiuc/224_224/training_label.npz')
