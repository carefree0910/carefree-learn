# type: ignore

import cflearn

from cflearn.misc.toolkit import check_is_ci


is_ci = check_is_ci()

data = cflearn.cv.MNISTData(
    root="../data",
    batch_size=4 if is_ci else 64,
    transform="to_tensor",
)

m = cflearn.cv.CarefreePipeline(
    "clf",
    {
        # "img_size": 28,
        "in_channels": 1,
        "num_classes": 10,
        "latent_dim": 512,
        "encoder1d": "backbone",
        "encoder1d_config": {"name": "resnet18"},
    },
    loss_name="cross_entropy",
    metric_names="acc",
    fixed_steps=1 if is_ci else None,
    valid_portion=0.0001 if is_ci else 1.0,
)
m.fit(data, cuda=None if is_ci else 0)
