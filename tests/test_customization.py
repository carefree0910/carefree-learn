import torch
import cflearn
import unittest

import numpy as np

from typing import Any
from typing import Optional

from cflearn.types import tensor_dict_type
from cflearn.protocol import TrainerState
from cflearn.models.ml import MLCoreProtocol


class TestCustomization(unittest.TestCase):
    def test_customize_model(self) -> None:
        @cflearn.ml.register_core("foo")
        class _(MLCoreProtocol):
            def __init__(self, in_dim: int, out_dim: int, num_history: int):
                super().__init__(in_dim, out_dim, num_history)
                self.dummy = torch.nn.Parameter(torch.tensor([1.0]))

            def forward(
                self,
                batch_idx: int,
                batch: tensor_dict_type,
                state: Optional[TrainerState] = None,
                **kwargs: Any
            ) -> tensor_dict_type:
                net = batch[cflearn.INPUT_KEY]
                net = net.new_empty([net.shape[0], 1]).fill_(1.0)
                return {cflearn.PREDICTIONS_KEY: net}

        x = np.random.random([1000, 10])
        y = np.random.random([1000, 1])
        m = cflearn.make(
            "ml.simple",
            config={
                "core_name": "foo",
                "output_dim": 1,
                "is_classification": False,
                "loss_name": "mae",
                "fixed_epoch": 0,
            },
        )
        m.fit(x, y)
        predictions = m.predict(x)[cflearn.PREDICTIONS_KEY]
        self.assertTrue(np.allclose(predictions, np.ones_like(y)))
        self.assertTrue(list(m.model.parameters())[0] is m.model.core.dummy)  # type: ignore


if __name__ == "__main__":
    unittest.main()
