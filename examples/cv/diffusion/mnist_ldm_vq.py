# type: ignore

import cflearn

from cflearn.data.blocks import *
from cflearn.zoo import load_predefined_config
from cflearn.toolkit import check_is_ci


is_ci = check_is_ci()


img_size = 64
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

d = load_predefined_config("diffusion/ldm")
d.model = "ddpm"
d.module_config["img_size"] = img_size // 4
d.module_config["in_channels"] = 3
d.module_config["out_channels"] = 3
d.module_config["start_channels"] = 32
d.module_config["attention_downsample_rates"] = [1, 2]
d.module_config["channel_multipliers"] = [1, 1, 2]
d.module_config["first_stage"] = "ae/vq.f4"
first_stage_config = load_predefined_config("ae/vq.f4").module_config
first_stage_config["img_size"] = img_size
first_stage_config["in_channels"] = 1
first_stage_config["out_channels"] = 1
first_stage_config["inner_channels"] = 32
first_stage_config["channel_multipliers"] = [1, 1, 2]
d.module_config["first_stage_config"] = first_stage_config
if is_ci:
    d.to_debug()
    d.module_config["sampler_config"] = {"default_steps": 1}


device = None if is_ci else 0
m = cflearn.DLTrainingPipeline.init(d).fit(data, device=device)
