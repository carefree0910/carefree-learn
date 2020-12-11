import os
import cflearn
import platform
import unittest

import numpy as np

from cftool.misc import shallow_copy_dict
from cfdata.tabular import TabularDataset

num_jobs = 0 if platform.system() == "Linux" else 2
logging_folder = "__test_dist__"
kwargs = {"fixed_epoch": 3}


class TestDist(unittest.TestCase):
    def test_experiment(self) -> None:
        x, y = TabularDataset.iris().xy
        exp_folder = os.path.join(logging_folder, "__test_experiment__")
        experiment = cflearn.Experiment(num_jobs=num_jobs)
        data_folder = experiment.dump_data_bundle(exp_folder, x, y)
        common_kwargs = {
            "root_workplace": exp_folder,
            "data_folder": data_folder,
            "config": kwargs,
        }
        experiment.add_task(model="fcnn", **shallow_copy_dict(common_kwargs))
        experiment.add_task(model="fcnn", **shallow_copy_dict(common_kwargs))
        experiment.add_task(model="tree_dnn", **shallow_copy_dict(common_kwargs))
        experiment.add_task(model="tree_dnn", **shallow_copy_dict(common_kwargs))
        results = experiment.run_tasks()
        ms = cflearn.load_experiment_results(results)
        saving_folder = os.path.join(logging_folder, "__test_experiment_save__")
        experiment.save(saving_folder)
        loaded = cflearn.Experiment.load(saving_folder)
        ms_loaded = cflearn.load_experiment_results(loaded.results)
        self.assertTrue(
            np.allclose(
                ms["fcnn"][1].predict(x),
                ms_loaded["fcnn"][1].predict(x),
            )
        )
        cflearn._rmtree(logging_folder)


if __name__ == "__main__":
    unittest.main()
