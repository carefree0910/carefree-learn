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

m = cflearn.api.ae_kl_f8(
    img_size,
    workplace="_kl",
    pretrained=False,
    debug=is_ci,
)
m.fit(data, cuda=None if is_ci else 1)
