import torch
import cflearn
import unittest

import numpy as np

from typing import Any


class TestCustomization(unittest.TestCase):
    def test_customize_model(self) -> None:
        @cflearn.register_extractor("foo_extractor")
        class _(cflearn.ExtractorBase):
            @property
            def out_dim(self) -> int:
                return 1

            def forward(self, net: torch.Tensor) -> torch.Tensor:
                return net.new_empty([net.shape[0], 1]).fill_(1.0)

        cflearn.register_config("foo_extractor", "default", config={})

        @cflearn.register_head("foo")
        class _(cflearn.HeadBase):  # type: ignore
            def __init__(self, in_dim: int, out_dim: int, **kwargs: Any):
                super().__init__(in_dim, out_dim, **kwargs)
                self.dummy = torch.nn.Parameter(torch.tensor([1.0]))

            def forward(self, net: torch.Tensor) -> torch.Tensor:
                return net

        cflearn.register_head_config("foo", "default", head_config={})

        x = np.random.random([1000, 10])
        y = np.random.random([1000, 1])
        pipe = cflearn.PipeInfo("foo", extractor="foo_extractor")
        cflearn.register_model("tce", pipes=[pipe])
        kwargs = {"task_type": "reg", "use_simplify_data": True, "fixed_epoch": 0}
        m = cflearn.make("tce", **kwargs).fit(x, y)  # type: ignore
        self.assertTrue(list(m.model.parameters())[0] is m.model.heads["foo"].dummy)  # type: ignore
        self.assertTrue(np.allclose(m.predict(x), np.ones_like(y)))  # type: ignore
        cflearn._rmtree("_logs")


if __name__ == "__main__":
    unittest.main()
