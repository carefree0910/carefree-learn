import os
import torch
import cflearn
import platform
import unittest

import numpy as np

from typing import Any
from torch.nn import Parameter
from cfdata.tabular import TabularDataset


IS_LINUX = platform.system() == "Linux"
file_folder = os.path.dirname(__file__)


class TestRegister(unittest.TestCase):
    def test_initializer(self) -> None:
        initializer = cflearn.Initializer()

        @cflearn.register_initializer("all_one")
        def all_one(_: Any, parameter: Parameter) -> None:
            parameter.fill_(1.0)

        n = 100
        param = Parameter(torch.zeros(n))
        with torch.no_grad():
            initializer.initialize(param, "all_one")

        self.assertTrue(np.allclose(param.data.numpy(), np.ones(n, np.float32)))

    def test_processor(self) -> None:
        @cflearn.register_processor("plus_one")
        class PlusOne(cflearn.Processor):
            @property
            def input_dim(self) -> int:
                return 1

            @property
            def output_dim(self) -> int:
                return 1

            def fit(self, columns: np.ndarray) -> cflearn.Processor:
                return self

            def _process(self, columns: np.ndarray) -> np.ndarray:
                return columns + 1

            def _recover(self, processed_columns: np.ndarray) -> np.ndarray:
                return processed_columns - 1

        logging_folder = "__test_register__"
        config = {
            "logging_folder": logging_folder,
            "data_config": {"label_process_method": "plus_one"},
        }
        toy = cflearn.make_toy_model(config=config)
        y = toy.data.converted.y
        processed_y = toy.data.processed.y
        self.assertTrue(np.allclose(y + 1, processed_y))
        cflearn._rmtree(logging_folder)

    def test_register_external(self) -> None:
        cflearn.remove_all_external()
        external_model = "external_linear"
        external_file = os.path.join(file_folder, "external_.py")
        cflearn.register_external(external_file)
        x, y = TabularDataset.iris().xy
        cflearn.make(external_model).fit(x, y)
        cflearn.repeat_with(x, y, models=external_model, num_repeat=3, num_jobs=1)
        if not IS_LINUX:
            cflearn.repeat_with(x, y, models=external_model, num_repeat=3, num_jobs=3)
        cflearn.remove_external(external_file)


if __name__ == "__main__":
    unittest.main()
