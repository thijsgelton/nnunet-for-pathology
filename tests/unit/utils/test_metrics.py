import unittest

import torch

from utils.metrics import Dice
import numpy as np


class TestCaseDiceMetric(unittest.TestCase):
    def test_something(self):
        num_classes = 3
        self.dice_metric = Dice(num_classes=num_classes)
        y_pred = torch.from_numpy(np.random.uniform(-2, 2, size=(1, num_classes, 256, 256)))
        y_true_argmax = np.random.uniform(-2, 2, size=(1, num_classes, 256, 256)).argmax(axis=1)
        y_true = torch.from_numpy(np.eye(3)[y_true_argmax].transpose(0, 3, 1, 2))

        self.dice_metric.update(y_pred, y_true)
        dice_score = self.dice_metric.compute(mean=True)
        assert 0.4 > dice_score.item() > 0.2  # Random weights with 3 classes should give a dice of 0.33


if __name__ == '__main__':
    unittest.main()
