import torch
import cflearn
import unittest

import numpy as np


class TestCustomization(unittest.TestCase):
    def test_customize_model(self) -> None:
        @cflearn.register_ml_module("foo")
        class _(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.tensor([1.0]))

            def forward(self, net: torch.Tensor) -> torch.Tensor:
                net = net.new_empty([net.shape[0], 1]).fill_(1.0)
                return {cflearn.PREDICTIONS_KEY: net}

        x = np.random.random([1000, 10])
        y = np.random.random([1000, 1])
        m = cflearn.api.make(
            "ml.simple",
            config={"core_name": "foo", "output_dim": 1, "fixed_epoch": 0},
        )
        data = cflearn.MLData(x, y, is_classification=False)
        m.fit(data)
        predictions = m.predict(data)[cflearn.PREDICTIONS_KEY]
        self.assertTrue(np.allclose(predictions, np.ones_like(y)))
        self.assertTrue(list(m.model.parameters())[0] is m.model.core.net.dummy)  # type: ignore


if __name__ == "__main__":
    unittest.main()
