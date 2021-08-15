# type: ignore

import cflearn
import argparse

# CI
parser = argparse.ArgumentParser()
parser.add_argument("--ci", type=int, default=0)
args = parser.parse_args()
is_ci = bool(args.ci)


train, valid = cflearn.cv.get_mnist(
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
        "encoder1d_configs": {"name": "resnet18"},
    },
    loss_name="cross_entropy",
    metric_names="acc",
    fixed_steps=1 if is_ci else None,
    valid_portion=0.0001 if is_ci else None,
)
m.fit(train, valid, cuda=None if is_ci else "0")
