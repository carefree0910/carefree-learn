import os
import cflearn

import numpy as np

from typing import *
from cftool.misc import shallow_copy_dict
from cflearn.models.ddr import DDRVisualizer


def test():
    power: int = 2  # Set to 5 will get much better results, but will also be very time consuming.
    verbose_level: int = 2
    anchor_ratios = quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    fit, check_cdf, check_quantile = True, True, True

    n = 10 ** power
    n_cv = int(min(10000, n * 0.1))
    export_folder = f"_logging/_pngs/power{power}"
    config: Dict[str, Any] = {
        "logging_file": f"power{power}.log",
        "verbose_level": verbose_level,
        "num_snapshot_per_epoch": 10,
        "batch_size": 128
    }

    def _get_file(folder, task_name):
        return None if folder is None else os.path.join(folder, task_name)

    def check(f, task_name):
        x = 2 * np.random.random([max(n, 2 * n_cv), 1]) - 1
        y = f(x)
        export_folder_ = os.path.join(export_folder, task_name)
        fetches = []
        x_min, x_max = x.min(), x.max()
        x_diff = x_max - x_min
        if check_cdf:
            fetches.append("cdf")
        if check_quantile:
            fetches.append("quantile")
        local_config = shallow_copy_dict(config)
        logging_folder = local_config["logging_folder"] = f"_logging/{task_name}"
        ddr_config = local_config.setdefault("ddr_config", {})
        ddr_config["fetches"] = fetches
        if export_folder_ is not None:
            os.makedirs(export_folder_, exist_ok=True)
        if not fit:
            m = cflearn.load(saving_folder=logging_folder)
        else:
            m = cflearn.make("ddr", **local_config).fit(x[n_cv:], y[n_cv:], x[:n_cv], y[:n_cv])
            m = cflearn.save(m, saving_folder=logging_folder)["ddr"]
        visualizer = DDRVisualizer(m)
        if check_cdf:
            export_path = _get_file(export_folder_, "cdf.png")
            visualizer.visualize(x, y, export_path, anchor_ratios=anchor_ratios)
        if check_quantile:
            export_path = _get_file(export_folder_, "quantile.png")
            visualizer.visualize(x, y, export_path, quantiles=quantiles)
        n_base, n_repeat = 1000, 10000
        x_base = np.linspace(x_min - 0.1 * x_diff, x_max + 0.1 * x_diff, n_base)[..., None]
        x_matrix = np.repeat(x_base, n_repeat, axis=1)
        y_matrix = f(x_matrix)
        if check_cdf:
            visualizer.visualize_multiple(
                x, y, x_base, y_matrix,
                export_folder_, anchor_ratios=anchor_ratios
            )
        if check_quantile:
            visualizer.visualize_multiple(
                x, y, x_base, y_matrix,
                export_folder_, quantiles=quantiles
            )

    check(lambda x: x + np.random.random(x.shape) * 3, "linear_constant")
    check(lambda x: x + np.random.random(x.shape) * 5 * x, "linear_linear")
    check(lambda x: x + 2 * x ** 2 + 3 * x * np.random.random(x.shape), "quad_linear")
    check(lambda x: np.sin(8 * x) + np.random.normal(0, 0.5 * np.ones_like(x)), "sin_constant")
    check(
        lambda x: (2 / (np.sqrt(3) * np.pi ** 0.25) * (1 - 25 * x ** 2) * np.exp(-12.5 * x ** 2))
                   + 0.5 * (np.random.random(x.shape) - 0.5), "mexican_hat_constant"
    )
    check(
        lambda x: (np.sin(0.5 * (x + 1) * np.pi)
                   + np.random.normal(0, np.exp(np.sin(np.pi * (x + 1))))), "complex_complex"
    )


if __name__ == '__main__':
    test()
