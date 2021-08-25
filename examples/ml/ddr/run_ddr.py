import os
import cflearn

import numpy as np


power = 4
padding = 1.0
output_folder = "fig"
ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

num_samples = 10 ** power
num_validation = num_test = int(min(10000, num_samples * 0.1))
num_samples = max(num_samples, 3 * num_validation)
num_train = num_samples - num_validation - num_test

f = lambda inp: (
    np.sin(0.5 * (inp + 1.0) * np.pi)
    + np.random.normal(0.0, np.exp(np.sin(np.pi * (inp + 1.0))))
)
x = 2.0 * np.random.random([num_samples, 1]) - 1.0
y = f(x)

data = cflearn.ml.MLData(
    x[:num_train],
    y[:num_train],
    x[num_train:-num_test],
    y[num_train:-num_test],
    is_classification=False,
)
x_test, y_test = x[-num_test:], y[-num_test:]

m = cflearn.ml.SimplePipeline("ddr", output_dim=1, loss_name="ddr")
m.fit(data)

os.makedirs(output_folder, exist_ok=True)
visualizer = cflearn.ml.DDRVisualizer(m.model.core)  # type: ignore
visualizer.visualize(
    x_test,
    y_test,
    os.path.join(output_folder, "all.png"),
    ratios=ratios,
    padding=padding,
)
visualizer.visualize_multiple(f, x_test, y_test, output_folder, ratios=ratios)
