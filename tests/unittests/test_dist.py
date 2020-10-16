import cflearn
import unittest

import numpy as np

from cfdata.tabular import TaskTypes
from cfdata.tabular import TabularDataset


kwargs = {"min_epoch": 1, "num_epoch": 2, "max_epoch": 4}


class TestDist(unittest.TestCase):
    def test_experiments(self) -> None:
        x, y = TabularDataset.iris().xy
        experiments = cflearn.Experiments("__test_experiments__")
        experiments.add_task(x, y, model="fcnn", **kwargs)  # type: ignore
        experiments.add_task(x, y, model="fcnn", **kwargs)  # type: ignore
        experiments.add_task(x, y, model="tree_dnn", **kwargs)  # type: ignore
        experiments.add_task(x, y, model="tree_dnn", **kwargs)  # type: ignore
        experiments.run_tasks(num_jobs=2)
        ms = cflearn.transform_experiments(experiments)
        saving_folder = "__test_experiments_save__"
        experiments.save(saving_folder)
        loaded = cflearn.Experiments.load(saving_folder)
        ms_loaded = cflearn.transform_experiments(loaded)
        self.assertTrue(
            np.allclose(ms["fcnn"][1].predict(x), ms_loaded["fcnn"][1].predict(x))
        )

    def test_benchmark(self) -> None:
        x, y = TabularDataset.iris().xy
        benchmark = cflearn.Benchmark(
            "foo",
            TaskTypes.CLASSIFICATION,
            models=["fcnn", "tree_dnn"],
            temp_folder="__test_benchmark__",
            increment_config=kwargs.copy(),
        )
        benchmarks = {
            "fcnn": {"default": {}, "sgd": {"optimizer": "sgd"}},
            "tree_dnn": {"default": {}, "adamw": {"optimizer": "adamw"}},
        }
        results = benchmark.k_fold(
            3,
            x,
            y,
            num_jobs=2,
            benchmarks=benchmarks,  # type: ignore
        )
        msg1 = results.comparer.log_statistics()
        saving_folder = "__test_benchmark_save__"
        benchmark.save(saving_folder)
        loaded_benchmark, loaded_results = cflearn.Benchmark.load(saving_folder)
        msg2 = loaded_results.comparer.log_statistics()
        self.assertEqual(msg1, msg2)


if __name__ == "__main__":
    unittest.main()
