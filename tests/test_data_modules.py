# type: ignore

import os
import torch
import cflearn
import unittest

import numpy as np

from PIL import Image
from typing import Set
from tempfile import TemporaryDirectory
from cftool.misc import Serializer
from cflearn.data import TorchData
from cflearn.schema import IData
from cflearn.constants import INPUT_KEY
from cflearn.data.blocks import DataOrder
from cflearn.data.blocks import DataSplitter
from cflearn.data.blocks import HWCToCHWBlock
from cflearn.data.blocks import ImageFolderBlock
from cflearn.data.blocks import ResizedPreparation


data_config = cflearn.DataConfig.inference_with(batch_size=4)


class TestDataModules(unittest.TestCase):
    def _test_data(
        self,
        key: str,
        data: IData,
        target_keys: Set[str],
    ) -> None:
        data_folder = os.path.join("_data", key)
        loader = data.get_loaders()[0]
        b1 = next(iter(loader))
        Serializer.save(data_folder, data)
        data = Serializer.load(data_folder, IData)
        loader = data.get_loaders()[0]
        b2 = next(iter(loader))
        for k, v in b1.items():
            self.assertTrue(np.allclose(v, b2[k]))
        diff = target_keys - set(b1.keys())
        self.assertEqual(len(diff), 0)

    def test_numpy_data(self) -> None:
        x_train = np.random.randn(10, 13)
        x_other1 = np.random.randn(10, 13)
        x_other2 = np.random.randn(10, 13)
        y_train = np.random.randn(10, 3)
        data = cflearn.ArrayData.init(config=data_config).fit(
            x_train,
            y_train,
            train_others={"other1": x_other1, "other2": x_other2},
        )
        self._test_data("numpy", data, {"input", "labels", "other1", "other2"})

    def test_numpy_dict_data(self) -> None:
        x_train = np.random.randn(10, 13)
        x_other1 = np.random.randn(10, 13)
        x_other2 = np.random.randn(10, 13)
        y_train = np.random.randn(10, 3)
        data = cflearn.ArrayDictData.init(config=data_config).fit(
            {"main": x_train, "other1": x_other1, "other2": x_other2}, y_train
        )
        self._test_data("numpy_dict", data, {"main", "labels", "other1", "other2"})

    def test_tensor_data(self) -> None:
        x_train = torch.rand(10, 13)
        x_other1 = torch.rand(10, 13)
        x_other2 = torch.rand(10, 13)
        y_train = torch.rand(10, 3)
        data = cflearn.ArrayData.init(config=data_config).fit(
            x_train,
            y_train,
            train_others={"other1": x_other1, "other2": x_other2},
        )
        self._test_data("tensor", data, {"input", "labels", "other1", "other2"})

    def test_tensor_dict_data(self) -> None:
        x_train = torch.rand(10, 13)
        x_other1 = torch.rand(10, 13)
        x_other2 = torch.rand(10, 13)
        y_train = torch.rand(10, 3)
        data = cflearn.ArrayDictData.init(config=data_config).fit(
            {"main": x_train, "other1": x_other1, "other2": x_other2}, y_train
        )
        self._test_data("tensor_dict", data, {"main", "labels", "other1", "other2"})

    def test_ml_data(self) -> None:
        x_train = np.random.randn(10, 13)
        y_train = np.random.randn(10, 3)
        data = cflearn.MLData.init(config=data_config).fit(x_train, y_train)
        self._test_data("ml", data, {"input", "labels"})

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

    def test_tensor_batcher(self) -> None:
        data: cflearn.ArrayDictData = cflearn.ArrayDictData.init()
        data.fit(dict(x1=torch.tensor([1]), x2=np.array(["foo"])))
        b = cflearn.TensorBatcher(data.get_loaders()[0], "cpu").get_full_batch()
        self.assertEqual(b["x1"].item(), 1)
        self.assertEqual(b["x2"].item(), "foo")

    def test_image_folder(self) -> None:
        num_samples = 5
        unified_size = 345
        with TemporaryDirectory() as dir:
            for i in range(num_samples):
                i_size = 100 * (i + 1)
                array = np.random.randint(0, 256, [i_size, i_size + 123, 3], np.uint8)
                Image.fromarray(array).save(os.path.join(dir, f"{i}.png"))
            image_folder_block = ImageFolderBlock(save_data_in_parallel=False)
            prep = ResizedPreparation(img_size=unified_size, keep_aspect_ratio=False)
            image_folder_block.set_preparation(prep)
            processor_config = cflearn.DataProcessorConfig()
            processor_config.set_blocks(image_folder_block, HWCToCHWBlock())
            data: TorchData = TorchData.build(dir, processor_config=processor_config)
            data.config.batch_size = 3
            train_loader, valid_loader = data.get_loaders()
            common_shapes = 3, unified_size, unified_size
            for i, batch in enumerate(train_loader):
                image = batch[INPUT_KEY]
                if i == 0:
                    self.assertSequenceEqual(image.shape, (3, *common_shapes))
                else:
                    self.assertSequenceEqual(image.shape, (1, *common_shapes))
            for i, batch in enumerate(valid_loader):  # type: ignore
                image = batch[INPUT_KEY]
                self.assertSequenceEqual(image.shape, (1, *common_shapes))


if __name__ == "__main__":
    unittest.main()
