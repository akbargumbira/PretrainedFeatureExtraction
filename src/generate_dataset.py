# coding=utf-8
import getpass
from src.utilities import (
    get_uiuc_training_data,
    save_dataset, data_path,
    DATASETS,
    generate_codalab_dataset,
    get_kaggle_dog_cat_data)


def generate_dataset(input_dir, target_size, output_dir, dataset,
                     reference_file=None, prefix='training'):
    """Generate ready to use dataset for 4 different datasets.

    :param dataset: 1 - UIUC, 2 - codalab gender and smile, 3 - kaggle dog
        and cat.
    :type dataset: int
    :return:
    """
    if dataset not in DATASETS:
        raise ValueError('Dataset should be 1, 2, or 3')

    if dataset == 1:
        training_id, training_data, training_label = get_uiuc_training_data(
            image_dir=input_dir, target_size=target_size)
        save_dataset(
            (training_id, training_data, training_label),
            output_dir
        )
    elif dataset == 2:
        generate_codalab_dataset(
            image_dir=input_dir,
            output_dir=output_dir,
            target_size=target_size,
            reference_file=reference_file,
            prefix=prefix)
    elif dataset == 3:
        training_id, training_data, training_label = get_kaggle_dog_cat_data(
            image_dir=input_dir, target_size=target_size)
        save_dataset(
            (training_id, training_data, training_label),
            output_dir
        )


target_size = (224, 224)

# UIUC
# Generate UIUC 224x224 Dataset
# image_dir = '/home/%s/dev/data/UIUC/event_img' % getpass.getuser()
# output_dir = data_path('uiuc', '224_224')
# generate_dataset(image_dir, target_size, output_dir, 1)
# -------------------------
# Codalab Training Set
# image_dir = '/home/%s/dev/data/codalab/smiles_trset' % getpass.getuser()
# output_dir = data_path('codalab', '224_224')
# generate_dataset(
#     image_dir,
#     target_size=(224, 224),
#     output_dir=output_dir,
#     dataset=2,
#     reference_file='gender_fex_trset.csv')

# Codalab Validation Set
# image_dir = '/home/%s/dev/data/codalab/smiles_valset' % getpass.getuser()
# output_dir = data_path('codalab', '224_224')
# generate_dataset(image_dir,
#                  target_size=(224, 224),
#                  output_dir=output_dir,
#                  dataset=2,
#                  reference_file='gender_fex_valset.csv',
#                  prefix='val')
# -------------------------
# Kaggle Dog and Cat Dataset
image_dir = '/home/%s/dev/data/kaggle_dog_cat/train' % getpass.getuser()
output_dir = data_path('kaggle_dog_cat', '224_224')
generate_dataset(image_dir,
                 target_size=(224, 224),
                 output_dir=output_dir,
                 dataset=3)
