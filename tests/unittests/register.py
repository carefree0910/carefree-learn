import torch
import cflearn
import unittest

import numpy as np

from torch.nn import Parameter


class TestRegister(unittest.TestCase):
    def test_initializer(self):
        initializer = cflearn.Initializer({})

        @cflearn.register_initializer("all_one")
        def all_one(initializer_, parameter):
            parameter.fill_(1.)

        n = 100
        param = Parameter(torch.zeros(n))
        with torch.no_grad():
            initializer.initialize(param, "all_one")

        self.assertTrue(np.allclose(param.data.numpy(), np.ones(n, np.float32)))

    def test_processor(self):

        @cflearn.register_processor("plus_one")
        class PlusOne(cflearn.Processor):
            @property
            def input_dim(self) -> int:
                return 1

            @property
            def output_dim(self) -> int:
                return 1

            def fit(self,
                    columns: np.ndarray) -> cflearn.Processor:
                return self

            def _process(self,
                         columns: np.ndarray) -> np.ndarray:
                return columns + 1

            def _recover(self,
                         processed_columns: np.ndarray) -> np.ndarray:
                return processed_columns - 1

        config = {"data_config": {"label_process_method": "plus_one"}}
        toy = cflearn.make_toy_model(config)
        y = toy.tr_data.converted.y
        processed_y = toy.tr_data.processed.y
        self.assertTrue(np.allclose(y + 1, processed_y))


if __name__ == '__main__':
    unittest.main()
