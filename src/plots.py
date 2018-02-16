# coding=utf-8
import itertools


import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(plt, cm, classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          rotation=0,
                          verbose=False):
    """Print and plot the confusion matrix.

    Parameters
    ----------
    cm : ConfusionMatrix
        Confusion matrix as created by ``sklearn.metrics.confusion_matrix``.

    classes : list(str), optional (default=None)
        List of class labels to be plotted as ticks in the matrix.

    normalize : bool, optional (default=False)
        Whether or not to normalize the results.

    title : str, optional (default='Confusion matrix')
        Title for the plot.

    cmap : matplotlib's colormap
        Colormap for the plot.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if classes is None:
        classes = np.arange(cm.shape[0])
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=rotation)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if verbose:
            print("Normalized confusion matrix")
    else:
        if verbose:
            print('Confusion matrix, without normalization')

    if verbose:
        print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
