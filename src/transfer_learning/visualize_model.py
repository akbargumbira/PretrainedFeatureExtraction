# coding=utf-8
import getpass


from matplotlib import pyplot as plt
import numpy as np
from quiver_engine import server

from src.utilities import root_path
from src.transfer_learning.train import get_prepared_model

# version = 3
# previous_model = root_path(
#     'src', 'transfer_learning', 'models', 'result_%s' % version, 'netbase',
#     'weights-last.hdf5')
#
# # Prepare the model
# model = get_prepared_model(10, version, previous_model, is_base=True)


from src.utilities import get_full_model, UIUC_EVENT_CLASS

# Visualize UIUC
base_model = 'resnet50'
image_size = (224, 224)
n_class = len(UIUC_EVENT_CLASS)
top_model_weights_path = 'uiuc/224_224/model/%s/checkpoint/weights' \
                         '-improvement-004-0.98.hdf5' % base_model
model = get_full_model(
    image_size,
    base_model,
    n_class,
    top_model_weights_path)

def make_conv_map(conv_weights,global_scale=False):
    shape = conv_weights.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows
    fig, axes = plt.subplots(nrows, ncols)

    if global_scale:
        wmin,wmax = np.min(conv_weights,(0,2,3)),np.max(conv_weights,(0,2,3)) #seperate min and max per channel

    for i,ax in enumerate(axes.flatten()):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        if i < shape[0]:
            w = conv_weights[i]
            if w.shape[0] > 3:
                w = np.sum(w, axis=0)[np.newaxis,:,:]
            w=np.squeeze(w.transpose(1,2,0))

            if not global_scale:
                wmin,wmax = np.min(w,(0,1)), np.max(w,(0,1)) #seperate min and max per channel

            w -= wmin
            w /= (wmax-wmin)
            ax.imshow(w, interpolation='nearest', cmap='gray')
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=1)
    plt.show()
    return fig, axes


make_conv_map(model.layers[1].get_weights()[0].transpose())
