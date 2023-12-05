# type: ignore

import cflearn

from cflearn.data.blocks import *
from cflearn.toolkit import check_is_ci


is_ci = check_is_ci()

data_config = cflearn.TorchDataConfig()
data_config.batch_size = 4 if is_ci else 64
processor_config = cflearn.DataProcessorConfig()
processor_config.set_blocks(
    TupleToBatchBlock(),
    ToNumpyBlock(),
    AffineNormalizeBlock(127.5, 127.5),
    HWCToCHWBlock(),
)
data = cflearn.mnist_data(data_config, processor_config)

config = cflearn.DLConfig(
    model="gan",
    module_name="gan",
    module_config=dict(
        img_size=28,
        in_channels=1,
        generator_config=dict(final_activation="tanh"),
    ),
)
if is_ci:
    config.to_debug()

device = None if is_ci else 0
m = cflearn.DLTrainingPipeline.init(config).fit(data, device=device)
