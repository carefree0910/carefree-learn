import os
import cflearn
import platform
import unittest

import numpy as np

from cftool.misc import shallow_copy_dict


num_jobs = 0 if platform.system() == "Linux" else 2
logging_folder = "__test_dist__"
kwargs = {"output_dim": 3, "fixed_epoch": 3}


class TestDist(unittest.TestCase):
    def test_experiment(self) -> None:
        x = np.random.randn(100, 10)
        y = np.random.randint(0, 3, [100, 1])
        exp_folder = os.path.join(logging_folder, "__test_experiment__")
        experiment = cflearn.dist.ml.Experiment(num_jobs=num_jobs)
        data = cflearn.MLData(x, y, is_classification=True)
        data.prepare(None)
        data_folder = experiment.dump_data(data, workplace=exp_folder)
        common_kwargs = {
            "root_workplace": exp_folder,
            "data_folder": data_folder,
            "config": kwargs,
        }
        experiment.add_task(model="fcnn", **shallow_copy_dict(common_kwargs))
        experiment.add_task(model="fcnn", **shallow_copy_dict(common_kwargs))
        experiment.add_task(model="linear", **shallow_copy_dict(common_kwargs))
        experiment.add_task(model="linear", **shallow_copy_dict(common_kwargs))
        results = experiment.run_tasks()
        load_results = cflearn.ml.load_experiment_results
        ms = load_results(results, cflearn.ml.MLSimplePipeline)
        saving_folder = os.path.join(logging_folder, "__test_experiment_save__")
        experiment.save(saving_folder)
        loaded = cflearn.dist.ml.Experiment.load(saving_folder)
        assert loaded.results is not None
        ms_loaded = load_results(loaded.results, cflearn.ml.MLSimplePipeline)
        idata = cflearn.MLInferenceData(x, cf_data=data.cf_data)
        self.assertTrue(
            np.allclose(
                ms["fcnn"][1].predict(idata)[cflearn.PREDICTIONS_KEY],
                ms_loaded["fcnn"][1].predict(idata)[cflearn.PREDICTIONS_KEY],
            )
        )


if __name__ == "__main__":
    unittest.main()
