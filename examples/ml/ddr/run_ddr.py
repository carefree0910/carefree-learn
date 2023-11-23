import os
import cflearn

import numpy as np

from cflearn.misc.toolkit import check_is_ci
from cflearn.misc.toolkit import seed_everything


seed_everything(123)

power = 4
padding = 1.0
output_folder = "fig"
ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

num_samples = 10**power
num_validation = num_test = int(min(10000, num_samples * 0.1))
num_samples = max(num_samples, 3 * num_validation)
num_train = num_samples - num_validation - num_test

f = lambda inp: (
    np.sin(0.5 * (inp + 1.0) * np.pi)
    + np.random.normal(0.0, np.exp(np.sin(np.pi * (inp + 1.0))))
)
x = 2.0 * np.random.random([num_samples, 1]) - 1.0
y = f(x)

m = cflearn.api.fit_ml(
    x[:num_train],
    y[:num_train],
    x[num_train:-num_test],
    y[num_train:-num_test],
    config=cflearn.MLConfig(
        model_name="ddr",
        model_config=dict(input_dim=1, output_dim=1),
    ),
    debug=check_is_ci(),
)

try:
    import matplotlib
except:
    from cftool.misc import print_warning

    print_warning(
        "`matplotlib` is not installed, "
        "so the visualization process will not be executed"
    )
    exit(0)

os.makedirs(output_folder, exist_ok=True)
visualizer = cflearn.ml.DDRVisualizer.from_pipeline(m)
x_test, y_test = x[-num_test:], y[-num_test:]
visualizer.visualize(
    x_test,
    y_test,
    os.path.join(output_folder, "all.png"),
    tau=ratios,
    padding=padding,
)
visualizer.visualize_multiple(f, x_test, y_test, output_folder, tau=ratios)
