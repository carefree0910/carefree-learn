import os
import cflearn
import platform
import unittest

import numpy as np

from cftool.misc import shallow_copy_dict
from cftool.misc import Serializer


num_jobs = 0 if platform.system() == "Linux" else 2
logging_folder = "__test_dist__"


class TestDist(unittest.TestCase):
    def test_experiment(self) -> None:
        x = np.random.randn(100, 10)
        y = np.random.randint(0, 3, [100, 1])
        config = cflearn.MLConfig(
            module_config=dict(input_dim=x.shape[1], output_dim=3),
            loss_name="focal",
            fixed_epoch=3,
        )
        exp_folder = os.path.join(logging_folder, "__test_experiment__")
        experiment = cflearn.dist.ml.Experiment(num_jobs=num_jobs)
        data = cflearn.MLData.init().fit(x, y)
        common_kwargs = dict(config=config, data=data, root_workspace=exp_folder)
        experiment.add_task(module="fcnn", **shallow_copy_dict(common_kwargs))
        experiment.add_task(module="fcnn", **shallow_copy_dict(common_kwargs))
        experiment.add_task(module="linear", **shallow_copy_dict(common_kwargs))
        experiment.add_task(module="linear", **shallow_copy_dict(common_kwargs))
        results = experiment.run_tasks()
        ms = cflearn.api.load_pipelines(results)
        saving_folder = os.path.join(logging_folder, "__test_experiment_save__")
        Serializer.save(saving_folder, experiment)
        loaded = Serializer.load(saving_folder, cflearn.dist.ml.Experiment)
        assert loaded.results is not None
        ms_loaded = cflearn.api.load_pipelines(loaded.results)
        loader = data.build_loader(x)
        self.assertTrue(
            np.allclose(
                ms["fcnn"][1].predict(loader)[cflearn.PREDICTIONS_KEY],
                ms_loaded["fcnn"][1].predict(loader)[cflearn.PREDICTIONS_KEY],
            )
        )


if __name__ == "__main__":
    unittest.main()
