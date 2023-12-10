# type: ignore

import cflearn

from cflearn.data.blocks import *
from cflearn.zoo import load_predefined_config
from cflearn.toolkit import check_is_ci


is_ci = check_is_ci()


img_size = 32
data_config = cflearn.TorchDataConfig()
data_config.batch_size = 4 if is_ci else 16
processor_config = cflearn.DataProcessorConfig()
processor_config.set_blocks(
    TupleToBatchBlock(),
    ToRGBBlock(),
    ToNumpyBlock(),
    ResizeBlock(img_size),
    AffineNormalizeBlock(127.5, 127.5),
    HWCToCHWBlock(),
)
data = cflearn.mnist_data(data_config, processor_config)

d = load_predefined_config("diffusion/ldm")
first_stage = "ae/kl.f4"
first_stage_config = load_predefined_config(first_stage).module_config
embedding_channels = first_stage_config["embedding_channels"]
d.model = "ddpm"
d.module_config["in_channels"] = embedding_channels
d.module_config["out_channels"] = embedding_channels
d.module_config["start_channels"] = 32
d.module_config["attention_downsample_rates"] = [1]
d.module_config["channel_multipliers"] = [1, 2]
d.module_config["first_stage"] = first_stage
first_stage_config["img_size"] = img_size
first_stage_config["pretrained"] = True
d.module_config["first_stage_config"] = first_stage_config
if is_ci:
    d.to_debug()
    first_stage_config["pretrained"] = False
    first_stage_config["inner_channels"] = 32
    first_stage_config["channel_multipliers"] = [1, 1, 2]
    d.module_config["sampler_config"] = {"default_steps": 1}

num_downsamples = len(first_stage_config["channel_multipliers"]) - 1
d.module_config["img_size"] = img_size // 2**num_downsamples

device = None if is_ci else 0
m = cflearn.DLTrainingPipeline.init(d).fit(data, device=device)
