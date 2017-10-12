# coding=utf-8
import os
import getpass
import numpy as np
from keras.preprocessing import image as keras_image
from utilities import load_dataset


def generate_dataset(image_dir, reference_file, target_size,
                     prefix='training', crop=False):
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
            print 'Procesed: %s %% of images' % (i * 100 / len(data))

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
                image = keras_image.load_img(image_path, target_size=target_size)
            image = keras_image.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            training_data.extend(image)

    print 'Saving the training id, data and label....'
    np.savez_compressed(open('model/%s_id.npz' % prefix, 'w'), id)
    np.savez_compressed(
        open('model/%s_data.npz' % prefix, 'w'), np.array(training_data))
    np.savez_compressed(
        open('model/%s_gender_label.npz' % prefix, 'w'), gender_labels)
    np.savez_compressed(
        open('model/%s_smile_label.npz' % prefix, 'w'), smile_labels)

# # Check
# image_dir = '/home/%s/dev/data/codalab/test_small' % getpass.getuser()
# generate_dataset(image_dir, 'gender_fex_valset.csv', target_size=(224, 224))

# Training
# image_dir = '/home/%s/dev/data/codalab/smiles_trset' % getpass.getuser()
# generate_dataset(image_dir, 'gender_fex_trset.csv', target_size=(150, 150))

# Validation
# image_dir = '/home/%s/dev/data/codalab/smiles_valset' % getpass.getuser()
# generate_dataset(image_dir, 'gender_fex_valset.csv', target_size=(150, 150),
#                  prefix='val')

# ---------------------------------CROP------------------
# # Check
# image_dir = '/home/%s/dev/data/codalab/test_small' % getpass.getuser()
# generate_dataset(image_dir, 'gender_fex_valset.csv', target_size=(50, 50),
#                  prefix='crop', crop=True)

# Training
# image_dir = '/home/%s/dev/data/codalab/smiles_trset' % getpass.getuser()
# generate_dataset(image_dir, 'gender_fex_trset.csv', target_size=(50, 50),
#                  prefix='crop_training', crop=True)

# Validation
image_dir = '/home/%s/dev/data/codalab/smiles_valset' % getpass.getuser()
generate_dataset(image_dir, 'gender_fex_valset.csv', target_size=(50, 50),
                 prefix='crop_val', crop=True)
