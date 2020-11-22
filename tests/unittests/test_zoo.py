import os
import cflearn
import platform
import unittest

from cfdata.tabular import TabularDataset

num_jobs = 0 if platform.system() == "Linux" else 2
logging_folder = "__test_zoo__"


class TestZoo(unittest.TestCase):
    @staticmethod
    def _test_zoo_core(model: str) -> None:
        x, y = TabularDataset.iris().xy
        zoo_folder = os.path.join(logging_folder, f"__{model}__")
        zoo = cflearn.Zoo(model)
        for key, config in zoo.benchmarks.items():
            local_logging_folder = os.path.join(zoo_folder, key)
            config["logging_folder"] = local_logging_folder
            m = cflearn.make(model, **config).fit(x, y)
            cflearn.evaluate(x, y, pipelines=m)
        cflearn._rmtree(logging_folder)

    def test_fcnn_zoo(self) -> None:
        self._test_zoo_core("fcnn")

    def test_tree_dnn_zoo(self) -> None:
        self._test_zoo_core("tree_dnn")


if __name__ == "__main__":
    unittest.main()
