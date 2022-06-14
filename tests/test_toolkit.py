import torch
import unittest

import numpy as np

from typing import Any
from typing import Dict
from cflearn.misc.toolkit import corr
from cflearn.misc.toolkit import to_standard
from cflearn.misc.toolkit import get_arguments
from cflearn.misc.toolkit import auto_num_layers
from cflearn.misc.toolkit import sort_dict_by_value


class TestToolkit(unittest.TestCase):
    def test_sort_dict_by_value(self) -> None:
        d = {"a": 2.0, "b": 1.0, "c": 3.0}
        self.assertSequenceEqual(list(sort_dict_by_value(d)), ["b", "a", "c"])

    def test_to_standard(self) -> None:
        def _check(src: np.dtype, tgt: np.dtype) -> None:
            self.assertEqual(to_standard(np.array([0], src)).dtype, tgt)

        _check(np.float16, np.float32)
        _check(np.float32, np.float32)
        _check(np.float64, np.float32)
        _check(np.int8, np.int64)
        _check(np.int16, np.int64)
        _check(np.int32, np.int64)
        _check(np.int64, np.int64)

    def test_get_arguments(self) -> None:
        def _1(a: int = 1, b: int = 2, c: int = 3) -> Dict[str, Any]:
            return get_arguments()

        class _2:
            def __init__(self, a: int = 1, b: int = 2, c: int = 3):
                self.kw = get_arguments()

        self.assertDictEqual(_1(), dict(a=1, b=2, c=3))
        self.assertDictEqual(_2().kw, dict(a=1, b=2, c=3))

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
        self.assertTrue(torch.allclose(corr00, corr10))
        self.assertTrue(torch.allclose(corr01, corr11))
        self.assertTrue(torch.allclose(corr01, corr02.t()))
        self.assertTrue(torch.allclose(corr11, corr12.t()))
        self.assertTrue(torch.allclose(corr02, corr12))


if __name__ == "__main__":
    unittest.main()
