# type: ignore

import cflearn

from cflearn.data.blocks import *
from cflearn.zoo import load_predefined_info
from cflearn.toolkit import check_is_ci


is_ci = check_is_ci()


img_size = 32
data_config = cflearn.TorchDataConfig()
data_config.batch_size = 4 if is_ci else 16
processor_config = cflearn.DataProcessorConfig()
processor_config.set_blocks(
    TupleToBatchBlock(),
    ToNumpyBlock(),
    ResizeBlock(img_size),
    AffineNormalizeBlock(127.5, 127.5),
    HWCToCHWBlock(),
)
data = cflearn.mnist_data(data_config, processor_config)

module_config = load_predefined_info("ae/vq.f4").module_config
module_config["img_size"] = img_size
module_config["in_channels"] = 1
module_config["out_channels"] = 1
module_config["inner_channels"] = 32
module_config["channel_multipliers"] = [1, 1, 2]
module_config["apply_tanh"] = True
config = cflearn.DLConfig(
    model="ae_vq",
    module_name="ae_vq",
    module_config=module_config,
)
if is_ci:
    config.to_debug()

device = None if is_ci else 0
m = cflearn.DLTrainingPipeline.init(config).fit(data, device=device)
