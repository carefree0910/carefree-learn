# type: ignore

import cflearn

from cflearn.data.blocks import *
from cflearn.toolkit import check_is_ci


is_ci = check_is_ci()

data_config = cflearn.TorchDataConfig()
data_config.batch_size = 4 if is_ci else 16
processor_config = cflearn.DataProcessorConfig()
processor_config.set_blocks(
    TupleToBatchBlock(),
    ToNumpyBlock(),
    AffineNormalizeBlock(127.5, 127.5),
    HWCToCHWBlock(),
)
data = cflearn.mnist_data(data_config, processor_config)

config = cflearn.DLConfig(
    model="vq_vae",
    module_name="vq_vae",
    module_config=dict(
        img_size=28,
        num_codes=32,
        in_channels=1,
        num_downsample=3,
        code_dimension=32 if is_ci else 256,
        num_classes=10,
        apply_tanh=True,
    ),
    callback_configs=dict(vq_vae=dict(num_classes=10)),
    loss_metrics_weights=dict(l2=1.0, commit=0.001),
    monitor_names="plateau",
)
if is_ci:
    config.to_debug()

device = None if is_ci else 0
m = cflearn.DLTrainingPipeline.init(config).fit(data, device=device)
