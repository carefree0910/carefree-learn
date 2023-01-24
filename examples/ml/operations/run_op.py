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

config = cflearn.MLConfig(
    output_dim=1,
    metric_names=metrics,
    tqdm_settings={"use_tqdm": True},
)
kwargs = {
    "data_config": dict(num_history=dim, is_classification=False),
    "config": config,
}
if is_ci:
    config.fixed_steps = 1

# add
config.core_name = "linear"
linear = cflearn.api.fit_ml(x, y_add, **kwargs)
config.core_name = "fcnn"
fcnn = cflearn.api.fit_ml(x, y_add, **kwargs)
config.core_name = "rnn"
rnn = cflearn.api.fit_ml(x, y_add, **kwargs)

try:
    import cfml

    idata = linear.make_inference_data(x, y_add)
    cflearn.ml.evaluate(idata, metrics=metrics, pipelines=[linear, fcnn, rnn])
except:
    from cftool.misc import print_warning

    print_warning(
        "`carefree-ml` is not installed, "
        "so the evaluation process will not be executed"
    )

linear_core = linear.model.core.core.net
print(f"w: {linear_core.weight.data}, b: {linear_core.bias.data}")  # type: ignore
