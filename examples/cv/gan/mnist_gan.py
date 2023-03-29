# type: ignore

import cflearn

from cflearn.data.blocks import *
from cflearn.misc.toolkit import check_is_ci


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
    model_name="gan",
    model_config=dict(img_size=28, in_channels=1),
)
if is_ci:
    config.to_debug()

cuda = None if is_ci else 0
m = cflearn.DLTrainingPipeline.init(config).fit(data, cuda=cuda)
