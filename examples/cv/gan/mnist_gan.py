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
    transform="for_generation",
)

m = cflearn.cv.CarefreePipeline(
    "gan",
    {"img_size": 28, "in_channels": 1},
    callback_names="generator",
    optimizer_settings={
        "g_parameters": {
            "optimizer": "adam",
            "scheduler": "warmup",
        },
        "d_parameters": {
            "optimizer": "adam",
            "scheduler": "warmup",
        },
    },
    fixed_steps=1 if is_ci else None,
    valid_portion=0.0001 if is_ci else None,
)
m.fit(train, valid, cuda=None if is_ci else "1")
