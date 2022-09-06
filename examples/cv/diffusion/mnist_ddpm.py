# type: ignore

import cflearn

from cflearn.misc.toolkit import check_is_ci


is_ci = check_is_ci()

img_size = 32
batch_size = 4 if is_ci else 16
data = cflearn.MNISTData(
    batch_size=batch_size,
    transform="ae_kl",
    test_transform="ae_kl_test",
    transform_config={"img_size": img_size},
)

model_config = {}
if is_ci:
    model_config["start_channels"] = 32
    model_config["sampler_config"] = {"default_steps": 1}

m = cflearn.api.ddpm(
    img_size,
    model_config=model_config,
    workplace="_ddpm",
    pretrained=False,
    debug=is_ci,
)
m.fit(data, cuda=None if is_ci else 2)
