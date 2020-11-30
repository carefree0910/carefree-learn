import os
import cflearn
import platform
import unittest

import numpy as np

from cfdata.tabular import TabularDataset

num_jobs = 0 if platform.system() == "Linux" else 2
logging_folder = "__test_dist__"
kwargs = {"fixed_epoch": 3}


class TestDist(unittest.TestCase):
    def test_experiments(self) -> None:
        x, y = TabularDataset.iris().xy
        exp_folder = os.path.join(logging_folder, "__test_experiments__")
        experiments = cflearn.Experiments(exp_folder)
        experiments.add_task(x, y, model="fcnn", **kwargs)  # type: ignore
        experiments.add_task(x, y, model="fcnn", **kwargs)  # type: ignore
        experiments.add_task(x, y, model="tree_dnn", **kwargs)  # type: ignore
        experiments.add_task(x, y, model="tree_dnn", **kwargs)  # type: ignore
        experiments.run_tasks(num_jobs=num_jobs)
        ms = cflearn.transform_experiments(experiments)
        saving_folder = os.path.join(logging_folder, "__test_experiments_save__")
        experiments.save(saving_folder)
        loaded = cflearn.Experiments.load(saving_folder)
        ms_loaded = cflearn.transform_experiments(loaded)
        self.assertTrue(
            np.allclose(
                ms["fcnn"][1].predict(x),
                ms_loaded["fcnn"][1].predict(x),
            )
        )
        cflearn._rmtree(logging_folder)


if __name__ == "__main__":
    unittest.main()
