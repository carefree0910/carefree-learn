import os
import cflearn

import numpy as np

from typing import *
from cftool.misc import shallow_copy_dict

from cflearn.pipeline import Pipeline
from cflearn.models.ddr.utils import DDRVisualizer


CI = True
power: int = 2
num_jobs: int = 1
padding: float = 1.0


def test() -> None:
    verbose_level: int = 2
    q_batch = np.array([0.1, 0.3, 0.5, 0.7, 0.9], np.float32)
    y_batch = np.array([0.1, 0.3, 0.5, 0.7, 0.9], np.float32)
    logging_folder = "__ddr__"

    n = 10 ** power
    n_cv = int(min(10000, n * 0.1))
    export_folder = f"{logging_folder}/_pngs/power{power}"
    config: Dict[str, Any] = {
        "logging_file": f"power{power}.log",
        "verbose_level": verbose_level,
        "num_snapshot_per_epoch": 10,
        "batch_size": 128,
    }
    if CI:
        config.update({"fixed_epoch": 3})

    info_dict = {}
    experiments = cflearn.Experiments(logging_folder)

    def _get_file(folder: str, task_name: str) -> Optional[str]:
        return None if folder is None else os.path.join(folder, task_name)

    def add_task(f: Callable, task_name: str) -> None:
        x = 2 * np.random.random([max(n, 2 * n_cv), 1]) - 1
        y = f(x)
        export_folder_ = os.path.join(export_folder, task_name)
        local_config = shallow_copy_dict(config)
        if CI:
            local_config["sample_weights"] = np.random.random(len(x))
        if export_folder_ is not None:
            os.makedirs(export_folder_, exist_ok=True)
        data_task = cflearn.Task.data_task(0, task_name, experiments)
        data_task.dump_data(x[n_cv:], y[n_cv:])
        data_task.dump_data(x[:n_cv], y[:n_cv], "_cv")
        if not CI:
            mlflow_config = None
        else:
            mlflow_config = {"task_name": task_name}
        experiments.add_task(
            model="ddr",
            identifier=task_name,
            mlflow_config=mlflow_config,
            data_task=data_task,
            **local_config,
        )
        info_dict[task_name] = {
            "f": f,
            "data_task": data_task,
            "export_folder": export_folder_,
        }

    def run_tasks() -> None:
        ms = experiments.run_tasks(
            num_jobs=num_jobs,
            load_task=cflearn.load_task,
        )
        for task_name, info in info_dict.items():
            m = ms[task_name][0]
            assert isinstance(m, Pipeline)
            f = info["f"]
            data_task = info["data_task"]
            export_folder_ = info["export_folder"]
            assert callable(f)
            assert isinstance(export_folder_, str)
            assert isinstance(data_task, cflearn.Task)
            x_cv, y_cv = data_task.fetch_data("_cv")
            assert isinstance(x_cv, np.ndarray)
            x_min, x_max = x_cv.min(), x_cv.max()
            x_diff = x_max - x_min
            visualizer = DDRVisualizer(m)
            # median residual
            export_path = _get_file(export_folder, "mr.png")
            visualizer.visualize(
                x_cv,
                y_cv,
                export_path,
                median_residual=True,
                padding=padding,
            )
            # quantile
            q_kwargs = {"q_batch": q_batch, "padding": padding}
            export_path = _get_file(export_folder, "quantile.png")
            visualizer.visualize(x_cv, y_cv, export_path, **q_kwargs)
            export_path = _get_file(export_folder, "med_mul.png")
            visualizer.visualize(x_cv, y_cv, export_path, mul_affine=True, **q_kwargs)
            # cdf
            y_kwargs = {"y_batch": y_batch, "padding": padding}
            cdf_path = _get_file(export_folder, "cdf.png")
            pdf_path = _get_file(export_folder, "pdf.png")
            visualizer.visualize(x_cv, y_cv, cdf_path, **y_kwargs)
            visualizer.visualize(x_cv, y_cv, pdf_path, to_pdf=True, **y_kwargs)
            export_path = _get_file(export_folder, "cdf_logit_mul.png")
            visualizer.visualize(
                x_cv,
                y_cv,
                export_path,
                cdf_logit_mul=True,
                **y_kwargs,
            )
            # multiple
            n_base, n_repeat = 1000, 10000
            x_base = np.linspace(x_min - 0.1 * x_diff, x_max + 0.1 * x_diff, n_base)
            x_base = x_base[..., None]
            x_matrix = np.repeat(x_base, n_repeat, axis=1)
            y_matrix = f(x_matrix)
            visualizer.visualize_multiple(
                x_cv,
                y_cv,
                x_base,
                y_matrix,
                export_folder,
                q_batch=q_batch,
            )
            visualizer.visualize_multiple(
                x_cv,
                y_cv,
                x_base,
                y_matrix,
                export_folder,
                y_batch=y_batch,
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
    if CI:
        cflearn._rmtree(logging_folder)


if __name__ == "__main__":
    test()
