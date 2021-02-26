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
    root_workplace = "__ddr__"

    n = 10 ** power
    n_cv = int(min(10000, n * 0.1))
    base_export_folder = f"{root_workplace}/_pngs/power{power}"
    config: Dict[str, Any] = {
        "logging_file": f"power{power}.log",
        "verbose_level": verbose_level,
        "num_snapshot_per_epoch": 10,
        "batch_size": 128,
    }
    if CI:
        config.update({"fixed_epoch": 3})

    info_dict = {}
    experiment = cflearn.Experiment(num_jobs=num_jobs)

    def _get_file(folder: str, task_name: str) -> Optional[str]:
        return None if folder is None else os.path.join(folder, task_name)

    def add_task(f: Callable, task_name: str) -> None:
        x = 2 * np.random.random([max(n, 2 * n_cv), 1]) - 1
        y = f(x)
        local_export_folder = os.path.join(base_export_folder, task_name)
        local_config = shallow_copy_dict(config)
        task_meta_kwargs = {}
        if CI:
            task_meta_kwargs["sample_weights"] = np.random.random(len(x))
        if local_export_folder is not None:
            os.makedirs(local_export_folder, exist_ok=True)
        if CI:
            local_config["mlflow_config"] = {"task_name": task_name}
        workplace = experiment.add_task(
            x[n_cv:],
            y[n_cv:],
            x[:n_cv],
            y[:n_cv],
            model="ddr",
            config=local_config,
            root_workplace=root_workplace,
            **task_meta_kwargs,
        )
        info_dict[task_name] = {
            "f": f,
            "workplace": workplace,
            "export_folder": local_export_folder,
        }

    def run_tasks() -> None:
        results = experiment.run_tasks(task_loader=cflearn.task_loader)
        pipeline_dict = results.pipeline_dict
        for task_name, info in info_dict.items():
            workplace = info["workplace"]
            local_export_folder = info["export_folder"]
            assert isinstance(workplace, str)
            assert isinstance(local_export_folder, str)
            f = info["f"]
            m = pipeline_dict[workplace]
            assert callable(f)
            assert isinstance(m, Pipeline)
            x_cv, y_cv = cflearn.Experiment.fetch_data("_cv", workplace=workplace)
            assert isinstance(x_cv, np.ndarray)
            x_min, x_max = x_cv.min(), x_cv.max()
            x_diff = x_max - x_min
            visualizer = DDRVisualizer(m)
            # median residual
            export_path = _get_file(local_export_folder, "mr.png")
            visualizer.visualize(
                x_cv,
                y_cv,
                export_path,
                median_residual=True,
                padding=padding,
            )
            # quantile
            q_kwargs = {"q_batch": q_batch, "padding": padding}
            export_path = _get_file(local_export_folder, "quantile.png")
            visualizer.visualize(x_cv, y_cv, export_path, **q_kwargs)
            export_path = _get_file(local_export_folder, "med_mul.png")
            visualizer.visualize(x_cv, y_cv, export_path, mul_affine=True, **q_kwargs)
            # cdf
            y_kwargs = {"y_batch": y_batch, "padding": padding}
            cdf_path = _get_file(local_export_folder, "cdf.png")
            pdf_path = _get_file(local_export_folder, "pdf.png")
            visualizer.visualize(x_cv, y_cv, cdf_path, **y_kwargs)
            visualizer.visualize(x_cv, y_cv, pdf_path, to_pdf=True, **y_kwargs)
            export_path = _get_file(local_export_folder, "cdf_logit_mul.png")
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
                local_export_folder,
                q_batch=q_batch,
            )
            visualizer.visualize_multiple(
                x_cv,
                y_cv,
                x_base,
                y_matrix,
                local_export_folder,
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
        cflearn._rmtree(root_workplace)
        cflearn._rmtree("mlruns")


if __name__ == "__main__":
    test()
