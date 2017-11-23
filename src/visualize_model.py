# coding=utf-8
import getpass

from quiver_engine import server

from src.utilities import get_full_model, UIUC_EVENT_CLASS

# Visualize UIUC
base_model = 'vgg16'
image_size = (224, 224)
n_class = len(UIUC_EVENT_CLASS)
top_model_weights_path = 'uiuc/224_224/model/%s/checkpoint/weights-improvement-178-0.97.hdf5' % base_model
image_dir = '/home/%s/dev/data/UIUC/test_small' % getpass.getuser()
temp_dir = '/home/%s/dev/data/UIUC/temp' % getpass.getuser()
model = get_full_model(
    image_size,
    base_model,
    n_class,
    top_model_weights_path)
server.launch(
    model,
    # classes=UIUC_EVENT_CLASS.keys(),
    input_folder=image_dir,
    temp_folder=temp_dir
)
