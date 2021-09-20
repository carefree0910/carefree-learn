# type: ignore

import cflearn

from cflearn.misc.toolkit import check_is_ci


is_ci = check_is_ci()

num_classes = 10
data = cflearn.cv.MNISTData(
    root="../data",
    batch_size=4 if is_ci else 64,
    transform="for_generation",
)

m = cflearn.cv.CarefreePipeline(
    "vq_vae",
    {
        "img_size": 28,
        "num_code": 16,
        "in_channels": 1,
        "target_downsample": 2,
        "num_classes": num_classes,
    },
    callback_configs={"vq_vae": {"num_classes": num_classes}},
    fixed_steps=1 if is_ci else None,
    valid_portion=0.0001 if is_ci else 1.0,
)
m.fit(data, cuda=None if is_ci else 1)
