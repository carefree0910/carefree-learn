import os
import cflearn

import numpy as np

from typing import *
from cftool.misc import shallow_copy_dict
from cflearn.models.ddr import DDRVisualizer


def test():
    power: int = 2
    num_jobs: int = 1
    verbose_level: int = 2
    anchor_ratios = quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    fit, check_cdf, check_quantile = True, True, True

    n = 10 ** power
    n_cv = int(min(10000, n * 0.1))
    export_folder = f"__tmp__/_pngs/power{power}"
    config: Dict[str, Any] = {
        "logging_file": f"power{power}.log",
        "verbose_level": verbose_level,
        "num_snapshot_per_epoch": 10,
        "batch_size": 128,
        "min_epoch": 1,
        "num_epoch": 2,
        "max_epoch": 4,
    }

    info_dict = {}
    experiments = cflearn.Experiments("__ddr__")

    def _get_file(folder, task_name):
        return None if folder is None else os.path.join(folder, task_name)

    def add_task(f, task_name):
        x = 2 * np.random.random([max(n, 2 * n_cv), 1]) - 1
        y = f(x)
        export_folder_ = os.path.join(export_folder, task_name)
        fetches = []
        if check_cdf:
            fetches.append("cdf")
        if check_quantile:
            fetches.append("quantile")
        local_config = shallow_copy_dict(config)
        local_config["sample_weights"] = np.random.random(len(x))
        ddr_config = local_config.setdefault("ddr_config", {})
        ddr_config["fetches"] = fetches
        if export_folder_ is not None:
            os.makedirs(export_folder_, exist_ok=True)
        data_task = cflearn.Task.data_task(0, task_name, experiments)
        data_task.dump_data(x[n_cv:], y[n_cv:])
        data_task.dump_data(x[:n_cv], y[:n_cv], "_cv")
        experiments.add_task(
            model="ddr",
            identifier=task_name,
            tracker_config={
                "project_name": "carefree-learn",
                "task_name": task_name,
                "overwrite": True,
            },
            data_task=data_task,
            **local_config,
        )
        info_dict[task_name] = {
            "f": f,
            "data_task": data_task,
            "export_folder": export_folder_,
        }

    def run_tasks():
        ms = experiments.run_tasks(
            num_jobs=num_jobs, run_tasks=fit, load_task=cflearn.load_task
        )
        for task_name, info in info_dict.items():
            m = ms[task_name][0]
            f = info["f"]
            data_task = info["data_task"]
            export_folder_ = info["export_folder"]
            x_cv, y_cv = data_task.fetch_data("_cv")
            x_min, x_max = x_cv.min(), x_cv.max()
            x_diff = x_max - x_min
            visualizer = DDRVisualizer(m)
            if check_cdf:
                export_path = _get_file(export_folder_, "cdf.png")
                visualizer.visualize(
                    x_cv, y_cv, export_path, anchor_ratios=anchor_ratios
                )
            if check_quantile:
                export_path = _get_file(export_folder_, "quantile.png")
                visualizer.visualize(x_cv, y_cv, export_path, quantiles=quantiles)
            n_base, n_repeat = 1000, 10000
            x_base = np.linspace(x_min - 0.1 * x_diff, x_max + 0.1 * x_diff, n_base)[
                ..., None
            ]
            x_matrix = np.repeat(x_base, n_repeat, axis=1)
            y_matrix = f(x_matrix)
            if check_cdf:
                visualizer.visualize_multiple(
                    x_cv,
                    y_cv,
                    x_base,
                    y_matrix,
                    export_folder_,
                    anchor_ratios=anchor_ratios,
                )
            if check_quantile:
                visualizer.visualize_multiple(
                    x_cv, y_cv, x_base, y_matrix, export_folder_, quantiles=quantiles
                )

    add_task(lambda x: x + np.random.random(x.shape) * 3, "linear_constant")
    add_task(lambda x: x + np.random.random(x.shape) * 5 * x, "linear_linear")
    add_task(
        lambda x: x + 2 * x ** 2 + 3 * x * np.random.random(x.shape), "quad_linear"
    )
    add_task(
        lambda x: np.sin(8 * x) + np.random.normal(0, 0.5 * np.ones_like(x)),
        "sin_constant",
    )
    add_task(
        lambda x: (
            2
            / (np.sqrt(3) * np.pi ** 0.25)
            * (1 - 25 * x ** 2)
            * np.exp(-12.5 * x ** 2)
        )
        + 0.5 * (np.random.random(x.shape) - 0.5),
        "mexican_hat_constant",
    )
    add_task(
        lambda x: (
            np.sin(0.5 * (x + 1) * np.pi)
            + np.random.normal(0, np.exp(np.sin(np.pi * (x + 1))))
        ),
        "complex_complex",
    )

    run_tasks()


if __name__ == "__main__":
    test()
