import cflearn
import unittest

import numpy as np

from cfdata.tabular import *


class TestDist(unittest.TestCase):
    def test_experiments(self):
        x, y = TabularDataset.iris().xy
        experiments = cflearn.Experiments("__test_experiments__")
        experiments.add_task(x, y, model="fcnn")
        experiments.add_task(x, y, model="fcnn")
        experiments.add_task(x, y, model="tree_dnn")
        experiments.add_task(x, y, model="tree_dnn")
        results = experiments.run_tasks(num_jobs=2)
        ms = {k: list(map(cflearn.load_task, v)) for k, v in results.items()}
        saving_folder = "__test_experiments_save__"
        experiments.save(saving_folder)
        loaded = cflearn.Experiments.load(saving_folder)
        ms_loaded = {k: list(map(cflearn.load_task, v)) for k, v in loaded.tasks.items()}
        self.assertTrue(np.allclose(ms["fcnn"][1].predict(x), ms_loaded["fcnn"][1].predict(x)))

    def test_benchmark(self):
        x, y = TabularDataset.iris().xy
        benchmark = cflearn.Benchmark(
            "foo",
            TaskTypes.CLASSIFICATION,
            models=["fcnn", "tree_dnn"],
            temp_folder="__test_benchmark__"
        )
        benchmarks = {
            "fcnn": {"default": {}, "sgd": {"optimizer": "sgd"}},
            "tree_dnn": {"default": {}, "adamw": {"optimizer": "adamw"}}
        }
        msg1 = benchmark.k_fold(3, x, y, num_jobs=2, benchmarks=benchmarks).comparer.log_statistics()
        saving_folder = "__test_benchmark_save__"
        benchmark.save(saving_folder)
        loaded_benchmark, loaded_results = cflearn.Benchmark.load(saving_folder)
        msg2 = loaded_results.comparer.log_statistics()
        self.assertEqual(msg1, msg2)


if __name__ == '__main__':
    unittest.main()
