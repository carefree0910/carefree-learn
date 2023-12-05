# type: ignore

import cflearn

from cflearn.toolkit import check_is_ci


is_ci = check_is_ci()

data_config = cflearn.TorchDataConfig()
data_config.batch_size = 4 if is_ci else 64
data = cflearn.mnist_data(data_config)

config = cflearn.DLConfig(
    module_name="cv_clf",
    module_config=dict(
        in_channels=1,
        num_classes=10,
        encoder_config=dict(num_downsample=3),
    ),
    loss_name="focal",
    metric_names="acc" if is_ci else ["acc", "auc"],
)
if is_ci:
    config.to_debug()

device = None if is_ci else 0
m = cflearn.MLTrainingPipeline.init(config).fit(data, device=device)
