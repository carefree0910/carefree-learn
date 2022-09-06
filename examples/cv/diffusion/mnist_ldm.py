# type: ignore

import cflearn

from cflearn.misc.toolkit import check_is_ci


is_ci = check_is_ci()

img_size = 256
latent_size = 32
batch_size = 4 if is_ci else 16

kw = {}
model_config = {}
if is_ci:
    img_size = 64
    latent_size = 2
    kw["latent_in_channels"] = 16
    kw["latent_out_channels"] = 16
    model_config["start_channels"] = 32
    model_config["channel_multipliers"] = (1,)
    model_config["sampler_config"] = {"default_steps": 1}
    model_config["first_stage"] = "ae/kl.f16"
    first_stage_config = model_config["first_stage_config"] = {}
    first_stage_config["pretrained"] = False

data = cflearn.MNISTData(
    batch_size=batch_size,
    transform="ae_kl",
    test_transform="ae_kl_test",
    transform_config={"img_size": img_size},
)

m = cflearn.api.ldm(
    latent_size,
    model_config=model_config,
    workplace="_ldm",
    pretrained=False,
    debug=is_ci,
    **kw,
)
m.fit(data, cuda=None if is_ci else 1)
