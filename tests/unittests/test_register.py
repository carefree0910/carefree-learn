import os
import torch
import cflearn
import platform
import unittest

import numpy as np

from typing import Any
from torch.nn import Parameter


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
        class _(cflearn.Processor):
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

        config = {"data_config": {"label_process_method": "plus_one"}}
        m = cflearn.ml.make_toy_model(config=config)
        assert isinstance(m, cflearn.ml.CarefreePipeline)
        y = m.data.converted.y
        processed_y = m.data.processed.y
        self.assertTrue(np.allclose(y + 1, processed_y))


if __name__ == "__main__":
    TestRegister().test_processor()
