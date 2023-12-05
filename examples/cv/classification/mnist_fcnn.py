# type: ignore

import cflearn

from cflearn.toolkit import check_is_ci


is_ci = check_is_ci()

data_config = cflearn.TorchDataConfig()
data_config.batch_size = 4 if is_ci else 64
data = cflearn.mnist_data(data_config, additional_blocks=[cflearn.FlattenBlock()])

config = cflearn.DLConfig(
    module_name="fcnn",
    module_config=dict(input_dim=784, output_dim=10),
    loss_name="focal",
    metric_names="acc" if is_ci else ["acc", "auc"],
)
if is_ci:
    config.to_debug()

device = None if is_ci else 0
m = cflearn.MLTrainingPipeline.init(config).fit(data, device=device)
