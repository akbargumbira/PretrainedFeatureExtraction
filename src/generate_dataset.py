# coding=utf-8
import getpass
from src.utilities import get_uiuc_training_data, save_dataset

image_dir = '/home/%s/dev/data/UIUC/event_img' % getpass.getuser()
target_size = (299, 299)
training_id, training_data, training_label = get_uiuc_training_data(
    image_dir=image_dir, target_size=target_size)
save_dataset(
    (training_id, training_data, training_label),
    'uiuc/%s_%s/' % (target_size[0], target_size[1])
)
