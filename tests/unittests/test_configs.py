import cflearn
import unittest

import numpy as np

from typing import Tuple
from cflearn.pipeline import Pipeline


class TestConfigs(unittest.TestCase):
    @staticmethod
    def _check_pruner(m: Pipeline) -> bool:
        return m.model.heads["fcnn"].mlp.mappings[0].linear.pruner is not None

    def _first_check(self, identifier: str) -> Tuple[np.ndarray, np.ndarray]:
        x = np.random.random([1000, 10])
        y = np.random.random([1000, 1])
        m = cflearn.make("fcnn", fixed_epoch=1, cuda="cpu").fit(x, y)
        self.assertTrue(self._check_pruner(m))
        cflearn.save(m, identifier)
        return x, y

    def _final_checks(self, identifier: str, x: np.ndarray, y: np.ndarray) -> None:
        m2 = cflearn.make("fcnn", fixed_epoch=1).fit(x, y)
        self.assertFalse(self._check_pruner(m2))
        m3 = cflearn.load(identifier)["fcnn"][0]
        self.assertTrue(self._check_pruner(m3))
        cflearn._remove(identifier)
        cflearn._rmtree("_logs")

    def test_preset(self) -> None:
        cflearn.ModelConfig("fcnn").switch().replace(head_config="pruned")
        identifier = "preset_pruned"
        x, y = self._first_check(identifier)
        cflearn.ModelConfig("fcnn").switch().replace(head_config="default")
        self._final_checks(identifier, x, y)

    def test_customize(self) -> None:
        head_config = cflearn.ModelConfig("fcnn").switch().head_config
        head_config["mapping_configs"] = {"pruner_config": {}}
        identifier = "customize_pruned"
        x, y = self._first_check(identifier)
        head_config.clear()
        self._final_checks(identifier, x, y)


if __name__ == "__main__":
    unittest.main()
