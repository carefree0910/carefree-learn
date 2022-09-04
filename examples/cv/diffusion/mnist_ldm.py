# type: ignore

import cflearn

from cflearn.misc.toolkit import check_is_ci


is_ci = check_is_ci()

img_size = 256
batch_size = 4 if is_ci else 16
data = cflearn.MNISTData(
    batch_size=batch_size,
    transform="ae_kl",
    test_transform="ae_kl_test",
    transform_config={"img_size": img_size},
)

m = cflearn.api.ldm(
    img_size,
    model_config=dict(
        start_channels=64,
    ),
    workplace="_ldm",
    pretrained=False,
    debug=is_ci,
)
m.fit(data, cuda=None if is_ci else 1)
