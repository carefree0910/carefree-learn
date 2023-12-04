# type: ignore

# This example requires the `mlflow` package

import cflearn

from cflearn.toolkit import check_is_ci
from cflearn.toolkit import seed_everything
from cflearn.data.ml import california_dataset


seed_everything(123)

x, y = california_dataset()
y = (y - y.mean()) / y.std()
config = cflearn.MLConfig(
    module_name="fcnn",
    module_config=dict(input_dim=x.shape[1], output_dim=1),
    loss_name="multi_task",
    loss_config=dict(loss_names=["mae", "mse"]),
    callback_names="mlflow",
)

block_names = ["ml_recognizer", "ml_preprocessor", "ml_splitter"]
m = cflearn.api.fit_ml(
    x,
    y,
    config=config,
    processor_config=cflearn.MLAdvancedProcessorConfig(block_names),
    debug=check_is_ci(),
)
loader = m.data.build_loader(x, y)
print("> metrics", m.evaluate(loader))

# After running the above codes, you should be able to
# see a `mlruns` folder in your current working dir.
# By executing `mlflow server`, you should be able to
# see those fancy metric curves (loss, lr, mae, mse,
# training loss, etc.) with a nice web interface
# at http://127.0.0.1:5000!
