import os
import torch
import cflearn
import unittest

import numpy as np

from typing import Set
from cflearn.data import DLDataModule
from cflearn.data.ml.utils import DataOrder
from cflearn.data.ml.utils import DataSplitter

try:
    from cfdata.tabular import TabularData
except:
    TabularData = None


class TestDataModules(unittest.TestCase):
    def _test_data(
        self,
        key: str,
        data: DLDataModule,
        target_keys: Set[str],
    ) -> None:
        data_folder = os.path.join("_data", key)
        loader = data.get_loaders()[0]
        b1 = next(iter(loader))
        data.save(data_folder)
        data = DLDataModule.load(data_folder)
        loader = data.initialize()[0]
        b2 = next(iter(loader))
        for k, v in b1.items():
            self.assertTrue(torch.allclose(v, b2[k]))
        diff = target_keys - set(b1.keys())
        self.assertEqual(len(diff), 0)

    def test_tensor_data(self) -> None:
        x_train = torch.rand(10, 13)
        x_other1 = torch.rand(10, 13)
        x_other2 = torch.rand(10, 13)
        y_train = torch.rand(10, 3)
        data = cflearn.TensorData(
            x_train,
            y_train,
            train_others={"other1": x_other1, "other2": x_other2},
            shuffle=False,
            batch_size=4,
        )
        self._test_data("tensor", data, {"input", "labels", "other1", "other2"})

    def test_tensor_dict_data(self) -> None:
        x_train = torch.rand(10, 13)
        x_other1 = torch.rand(10, 13)
        x_other2 = torch.rand(10, 13)
        y_train = torch.rand(10, 3)
        data = cflearn.TensorDictData(
            {"main": x_train, "other1": x_other1, "other2": x_other2},
            y_train,
            shuffle=False,
            batch_size=4,
        )
        self._test_data("tensor_dict", data, {"main", "labels", "other1", "other2"})

    def test_ml_data(self) -> None:
        x_train = np.random.randn(10, 13)
        y_train = np.random.randn(10, 3)
        data = cflearn.MLData(
            x_train,
            y_train,
            shuffle_train=False,
            is_classification=False,
            batch_size=4,
        )
        self._test_data("ml", data, {"input", "labels"})

    def test_ml_cf_data(self) -> None:
        if TabularData is None:
            return
        x_train = np.random.randn(10, 13)
        y_train = np.random.randn(10, 1)
        data = cflearn.MLCarefreeData(
            x_train,
            y_train,
            shuffle_train=False,
            batch_size=4,
        )
        self._test_data("ml_cf", data, {"input", "labels"})

    def test_ml_data_splitter(self) -> None:
        x = np.arange(12).reshape([6, 2])
        y = np.zeros(6, int)
        y[[-1, -2]] = 1
        splitter = DataSplitter()

        result = splitter.split(x, y, 3)
        self.assertListEqual(result.y.ravel().tolist(), [0, 0, 1])
        result = splitter.split(x, y, 0.5)
        self.assertListEqual(result.y.ravel().tolist(), [0, 0, 1])
        splitter.order = DataOrder.TOP_DOWN
        result = splitter.split(x, y, 3)
        self.assertListEqual(result.x.tolist(), [[0, 1], [2, 3], [8, 9]])
        self.assertListEqual(result.x_remained.tolist(), [[4, 5], [6, 7], [10, 11]])
        result = splitter.split(x, y, 0.5)
        self.assertListEqual(result.x.tolist(), [[0, 1], [2, 3], [8, 9]])
        self.assertListEqual(result.x_remained.tolist(), [[4, 5], [6, 7], [10, 11]])
        splitter.order = DataOrder.BOTTOM_UP
        result = splitter.split(x, y, 3)
        self.assertListEqual(result.x.tolist(), [[6, 7], [4, 5], [10, 11]])
        self.assertListEqual(result.x_remained.tolist(), [[2, 3], [0, 1], [8, 9]])
        result = splitter.split(x, y, 0.5)
        self.assertListEqual(result.x.tolist(), [[6, 7], [4, 5], [10, 11]])
        self.assertListEqual(result.x_remained.tolist(), [[2, 3], [0, 1], [8, 9]])

        y[-2] = 0
        splitter.shuffle = True
        result = splitter.split(x, y, 2)
        self.assertListEqual(result.y.ravel().tolist(), [0, 1])
        self.assertListEqual(result.y_remained.ravel().tolist(), [0, 0, 0, 0, 1])
        splitter.order = DataOrder.BOTTOM_UP
        result = splitter.split(x, y, 2)
        self.assertListEqual(result.x.tolist(), [[8, 9], [10, 11]])
        self.assertListEqual(
            result.x_remained.tolist(),
            [[6, 7], [4, 5], [2, 3], [0, 1], [10, 11]],
        )
        splitter.order = DataOrder.TOP_DOWN
        result = splitter.split(x, y, 2)
        self.assertListEqual(result.x.tolist(), [[0, 1], [10, 11]])
        self.assertListEqual(
            result.x_remained.tolist(),
            [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        )


if __name__ == "__main__":
    unittest.main()
