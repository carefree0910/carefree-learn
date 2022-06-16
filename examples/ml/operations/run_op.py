# type: ignore

import torch
import cflearn

import numpy as np

from cflearn.misc.toolkit import check_is_ci


is_ci = check_is_ci()

# for reproduction
np.random.seed(142857)
torch.manual_seed(142857)

# prepare
dim = 5
num_data = 10000
metrics = ["mae", "mse"]

x = np.random.random([num_data, dim, 1]) * 2.0
y_add = np.sum(x, axis=1)
y_prod = np.prod(x, axis=1)
y_mix = np.hstack([y_add, y_prod])

kwargs = {
    "data_config": dict(num_history=dim, is_classification=False),
    "output_dim": 1,
    "metric_names": metrics,
    "tqdm_settings": {"use_tqdm": True},
}
if is_ci:
    kwargs["fixed_steps"] = 1

# add
linear = cflearn.api.fit_ml(x, y_add, core_name="linear", **kwargs)
fcnn = cflearn.api.fit_ml(x, y_add, core_name="fcnn", **kwargs)
rnn = cflearn.api.fit_ml(x, y_add, core_name="rnn", **kwargs)
cflearn.ml.evaluate(linear.data, metrics=metrics, pipelines=[linear, fcnn, rnn])

linear_core = linear.model.core.core.net
print(f"w: {linear_core.weight.data}, b: {linear_core.bias.data}")  # type: ignore
