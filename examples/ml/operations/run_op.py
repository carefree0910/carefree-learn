# type: ignore

import cflearn

import numpy as np

from cflearn.misc.toolkit import check_is_ci
from cflearn.misc.toolkit import seed_everything


is_ci = check_is_ci()
seed_everything(123)

# prepare
dim = 5
num_data = 10000
metrics = ["mae", "mse"]

x = np.random.random([num_data, dim, 1]) * 2.0
y_add = np.sum(x, axis=1)
y_prod = np.prod(x, axis=1)
y_mix = np.hstack([y_add, y_prod])

config = cflearn.MLConfig(
    model_config=dict(input_dim=1, output_dim=1, num_history=dim),
    loss_name="multi_task",
    loss_config=dict(loss_names=["mae", "mse"]),
    metric_names=metrics,
    max_epoch=200,
    tqdm_settings={"use_tqdm": True},
)
kw = dict(
    config=config,
    processor_config=cflearn.MLAdvancedProcessorConfig(),
    debug=is_ci,
)

# add
config.model_name = "linear"
linear = cflearn.api.fit_ml(x, y_add, **kw)
config.model_name = "fcnn"
fcnn = cflearn.api.fit_ml(x, y_add, **kw)
config.model_name = "rnn"
rnn = cflearn.api.fit_ml(x, y_add, cuda=None if is_ci else 0, **kw)

# evaluate
cflearn.api.evaluate(
    linear.data.build_loader(x, y_add),
    dict(linear=linear, fcnn=fcnn, rnn=rnn),
)

linear_model: cflearn.Linear = linear.build_model.model
print(f"w: {linear_model.net.weight.data}, b: {linear_model.net.bias.data}")
