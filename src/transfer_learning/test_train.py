# coding=utf-8
import unittest

import numpy as np

from src.transfer_learning.train import get_prepared_model, get_proper_position
from src.utilities import root_path


class TestTrain(unittest.TestCase):
    """Test routines in train.py module."""
    def test_get_prepared_model(self):
        previous_model = root_path(
            'src', 'transfer_learning', 'models', 'result', 'half_rand0A',
            'weights-last.hdf5')
        model = get_prepared_model(5, previous_model)
        layer_idx = [0, 1, 4, 5, 8, 9, 13]
        for i in range(1, 7):
            layer_pos = get_proper_position(i)
            test_model = get_prepared_model(
                5, previous_model, copied_pos=i, copied_weight_trainable=False)
            # The weights from layer 0 - layer_pos should be the same. From
            # layer_post + 1 to len(layers) should be different
            for j in range(layer_pos+1):
                if j in layer_idx:
                    # the kernel
                    self.assertTrue(
                        np.array_equal(
                            model.layers[j].get_weights()[0],
                            test_model.layers[j].get_weights()[0]
                        ))
                    # the bias
                    self.assertTrue(
                        np.array_equal(
                            model.layers[j].get_weights()[1],
                            test_model.layers[j].get_weights()[1]
                        ))
                    self.assertFalse(test_model.layers[j].trainable)

            n_layers = len(model.layers)
            for j in range(layer_pos+1, n_layers):
                if j in layer_idx:
                    # the kernel
                    self.assertFalse(
                        np.array_equal(
                            model.layers[j].get_weights()[0],
                            test_model.layers[j].get_weights()[0]
                        ))
                    # the bias
                    self.assertFalse(
                        np.array_equal(
                            model.layers[j].get_weights()[1],
                            test_model.layers[j].get_weights()[1]
                        ))
                    self.assertTrue(test_model.layers[j].trainable)
