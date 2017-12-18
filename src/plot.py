# coding=utf-8
import matplotlib.pyplot as plt

from src.utilities import load_serialized_object, root_path


def plot_history(hist_path):
    hist = load_serialized_object(hist_path)

    plt.plot(range(1, 1 + len(hist['acc'])), hist['acc'], label='Acc',
             color='r')
    plt.plot(range(1, 1 + len(hist['val_acc'])), hist['val_acc'],
             label='Val Acc', linestyle='--', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
