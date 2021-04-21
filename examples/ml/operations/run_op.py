import torch
import cflearn
import argparse

import numpy as np


# CI
parser = argparse.ArgumentParser()
parser.add_argument("--ci", type=int, default=0)
args = parser.parse_args()
is_ci = bool(args.ci)

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
    "output_dim": 1,
    "is_classification": False,
    "loss_name": "mae",
    "num_history": dim,
    "metric_names": metrics,
    "tqdm_settings": {"use_tqdm": True},
}
if is_ci:
    kwargs["num_epoch"] = 3
    kwargs["max_epoch"] = 3

print(kwargs)

# add
linear = cflearn.ml.SimplePipeline("linear", **kwargs)  # type: ignore
fcnn = cflearn.ml.SimplePipeline("fcnn", **kwargs)  # type: ignore
rnn = cflearn.ml.SimplePipeline("rnn", **kwargs)  # type: ignore
linear.fit(x, y_add)
fcnn.fit(x, y_add)
rnn.fit(x, y_add)
cflearn.ml.evaluate(x, y_add, metrics=metrics, pipelines=[linear, fcnn, rnn])

linear_core = linear.model.core.net
print(f"w: {linear_core.weight.data}, b: {linear_core.bias.data}")  # type: ignore
