import torch
import unittest

from cftool.array import corr
from cftool.array import allclose
from cflearn.misc.toolkit import auto_num_layers


class TestToolkit(unittest.TestCase):
    def test_auto_num_layers(self) -> None:
        for img_size in [3, 7, 11, 23, 37, 53]:
            for min_size in [1, 2, 4, 8]:
                if min_size > img_size:
                    continue
                num_layers = auto_num_layers(img_size, min_size, None)
                if num_layers == 0:
                    self.assertTrue(img_size < 2 * min_size)

    def test_corr(self) -> None:
        pred = torch.randn(100, 5)
        target = torch.randn(100, 5)
        weights = torch.zeros(100, 1)
        weights[:30] = weights[-30:] = 1.0
        corr00 = corr(pred, pred, weights)
        corr01 = corr(pred, target, weights)
        corr02 = corr(target, pred, weights)
        w_pred = pred[list(range(30)) + list(range(70, 100))]
        w_target = target[list(range(30)) + list(range(70, 100))]
        corr10 = corr(w_pred, w_pred)
        corr11 = corr(w_pred, w_target)
        corr12 = corr(w_target, w_pred)
        self.assertTrue(allclose(corr00, corr10))
        self.assertTrue(allclose(corr01, corr11, corr02.t(), corr12.t()))


if __name__ == "__main__":
    unittest.main()
