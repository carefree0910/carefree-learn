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
data_kwargs = dict(num_history=dim, is_classification=False)
add_data = cflearn.ml.MLData(x, y_add, **data_kwargs)
prod_data = cflearn.ml.MLData(x, y_prod, **data_kwargs)
mix_data = cflearn.ml.MLData(x, y_mix, **data_kwargs)

kwargs = {
    "output_dim": 1,
    "metric_names": metrics,
    "tqdm_settings": {"use_tqdm": True},
}
if is_ci:
    kwargs["num_epoch"] = 3
    kwargs["max_epoch"] = 3

# add
linear = cflearn.ml.SimplePipeline("linear", **kwargs)  # type: ignore
fcnn = cflearn.ml.SimplePipeline("fcnn", **kwargs)  # type: ignore
rnn = cflearn.ml.SimplePipeline("rnn", **kwargs)  # type: ignore
linear.fit(add_data)
fcnn.fit(add_data)
rnn.fit(add_data)
cflearn.ml.evaluate(add_data, metrics=metrics, pipelines=[linear, fcnn, rnn])

linear_core = linear.model.core.net
print(f"w: {linear_core.weight.data}, b: {linear_core.bias.data}")  # type: ignore
