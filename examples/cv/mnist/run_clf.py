import torch
import cflearn

import numpy as np
import torch.nn as nn

from cflearn.misc.toolkit import check_is_ci
from cflearn.misc.toolkit import inject_debug


# preparations
is_ci = check_is_ci()
data = cflearn.cv.MNISTData(batch_size=16, transform="to_tensor")
# for reproduction
np.random.seed(142857)
torch.manual_seed(142857)


@cflearn.register_module("simple_conv")
class SimpleConvClassifier(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(128, num_classes),
        )


kwargs = dict(
    loss_name="cross_entropy",
    metric_names="acc" if is_ci else ["acc", "auc"],
    cuda=None if is_ci else 0,
)
if is_ci:
    inject_debug(kwargs)

cflearn.api.fit_cv(
    data,
    "simple_conv",
    {"in_channels": 1, "num_classes": 10},
    **kwargs,
)
