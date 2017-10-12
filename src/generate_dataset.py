# coding=utf-8
import getpass
from src.utilities import get_uiuc_training_data, save_dataset

image_dir = '/home/%s/dev/data/UIUC/test_small' % getpass.getuser()
training_id, training_data, training_label = get_uiuc_training_data(
    image_dir=image_dir, target_size=(224, 224))
save_dataset(
    (training_id, training_data, training_label),
    'uiuc/224_224/'
)
