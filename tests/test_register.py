import os
import torch
import cflearn
import platform
import unittest

import numpy as np

from typing import Any
from torch.nn import Parameter


class TestRegister(unittest.TestCase):
    def test_initializer(self) -> None:
        initializer = cflearn.Initializer()

        @cflearn.register_initializer("all_one")
        def _(_: Any, parameter: Parameter) -> None:
            parameter.fill_(1.0)

        n = 100
        param = Parameter(torch.zeros(n))
        with torch.no_grad():
            initializer.initialize(param, "all_one")

        self.assertTrue(np.allclose(param.data.numpy(), np.ones(n, np.float32)))


if __name__ == "__main__":
    unittest.main()
