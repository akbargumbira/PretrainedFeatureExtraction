# coding=utf-8
import copy
from keras import applications
import numpy as np
from sklearn.metrics.classification import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from src.utilities import get_full_model, load_dataset_from_path, CODALAB_GENDER_CLASS
from src.plots import plot_confusion_matrix


class TLClassifier(object):
    def __init__(self, base_arc, n_class, weights_path):
        self._image_size = (224, 224)
        self._base_arc = base_arc
        self._n_class = n_class
        self._weights_path = weights_path
        self._model = get_full_model(
            self._image_size, self._base_arc, self._n_class, self._weights_path)

        if self._base_arc == 'vgg16':
            self._preprocess_input = applications.vgg16.preprocess_input
        elif self._base_arc == 'inceptionv3':
            self._preprocess_input = applications.inception_v3.preprocess_input
        elif self._base_arc == 'resnet50':
            self._preprocess_input = applications.resnet50.preprocess_input
        else:
            raise ValueError('Model available: vgg16, inception_v3, resnet50')

    def predict(self, x,
                batch_size=None,
                verbose=0):
        x_cp = copy.deepcopy(x)
        self._preprocess_input(x_cp)
        return self._model.predict(x_cp, batch_size, verbose)

    def evaluate_metrics(self, y, y_true, metrics):
        scores = []
        for metric in metrics:
            scores.append(metric(y_true, y))
        return scores

if __name__ == 'main':
    # # Codalab Gender 3 classes
    # # VGG16 bottleneck path
    # vgg16_model = TLClassifier('vgg16', 3, 'codalab/224_224/model/gender/vgg16/checkpoint/weights-improvement-189-0.82.hdf5')
    # inception_model = TLClassifier('inceptionv3', 3, 'codalab/224_224/model/gender/inceptionv3/checkpoint/weights-improvement-132-0.82.hdf5')
    # resnet50_model = TLClassifier('resnet50', 3, 'codalab/224_224/model/gender/resnet50/checkpoint/weights-improvement-014-0.82.hdf5')
    #
    # val_data_path = 'codalab/224_224/val_data.npz'
    # val_label_path = 'codalab/224_224/val_gender_label.npz'
    # val_data = load_dataset_from_path(val_data_path)[:2]
    # val_label = load_dataset_from_path(val_label_path)[:2]
    # metrics = [accuracy_score, confusion_matrix]
    #
    # vgg16_preds = vgg16_model.predict(val_data)
    # vgg16_y = np.argmax(vgg16_preds, axis=1)
    # vgg16_scores = vgg16_model.evaluate_metrics(vgg16_y, val_label, metrics)
    #
    # inception_preds = inception_model.predict(val_data)
    # inception_y = np.argmax(inception_preds, axis=1)
    # inception_scores = inception_model.evaluate_metrics(inception_y, val_label, metrics)
    #
    # resnet50_preds = resnet50_model.predict(val_data)
    # resnet50_y = np.argmax(resnet50_preds, axis=1)
    # resnet50_scores = resnet50_model.evaluate_metrics(resnet50_y, val_label, metrics)
    #
    # plot_confusion_matrix(plt, vgg16_scores[1], CODALAB_GENDER_CLASS.values())

    # Codalab Smile 2 classes
    # VGG16 bottleneck path
    vgg16_model = TLClassifier('vgg16', 2,
                               'codalab/224_224/model/smile/vgg16/checkpoint/improved-190-0.74.hdf5')
    inception_model = TLClassifier('inceptionv3', 2,
                                   'codalab/224_224/model/smile/inceptionv3/checkpoint/improved-001-0.64.hdf5')
    resnet50_model = TLClassifier('resnet50', 2,
                                  'codalab/224_224/model/smile/resnet50/checkpoint/improved-017-0.75.hdf5')

    val_data_path = 'codalab/224_224/val_data.npz'
    val_label_path = 'codalab/224_224/val_smile_label.npz'
    val_data = load_dataset_from_path(val_data_path)
    val_label = load_dataset_from_path(val_label_path)
    metrics = [accuracy_score, confusion_matrix]

    vgg16_preds = vgg16_model.predict(val_data)
    vgg16_y = np.argmax(vgg16_preds, axis=1)
    vgg16_scores = vgg16_model.evaluate_metrics(vgg16_y, val_label, metrics)

    inception_preds = inception_model.predict(val_data)
    inception_y = np.argmax(inception_preds, axis=1)
    inception_scores = inception_model.evaluate_metrics(inception_y, val_label,
                                                        metrics)

    resnet50_preds = resnet50_model.predict(val_data)
    resnet50_y = np.argmax(resnet50_preds, axis=1)
    resnet50_scores = resnet50_model.evaluate_metrics(resnet50_y, val_label,
                                                      metrics)
