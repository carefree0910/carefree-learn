import torch

import numpy as np

from typing import Optional

from ...data import MLData
from ...models import CommonMLModel
from ...schema import data_type
from ...toolkit import get_dtype
from ...toolkit import get_device
from ...toolkit import show_or_save

try:
    import matplotlib.pyplot as plt
except:
    plt = Figure = None
try:
    from captum.attr import IntegratedGradients
except:
    IntegratedGradients = None


class Interpreter:
    def __init__(self, data: MLData, model: CommonMLModel) -> None:
        if IntegratedGradients is None:
            raise ValueError("`captum` is not required for `Interpreter`")
        if plt is None:
            raise ValueError("`matplotlib` is not required for `Interpreter`")
        self.data = data
        self.model = model
        self.encoder = model.get_encoder()

    def interpret(
        self,
        x: data_type,
        *,
        title: str = "Average Feature Importances",
        axis_title: str = "Features",
        export_path: Optional[str] = None,
    ) -> None:
        m = self.model.m
        x = self.data.build_loader(x).dataset.x
        x = torch.from_numpy(x).to(get_device(m), get_dtype(m))
        ig = IntegratedGradients(self.model)
        attr, delta = ig.attribute(x, return_convergence_delta=True)
        feature_names = self.data.feature_header
        x_pos = np.arange(len(feature_names))
        importances = attr.mean(axis=0)
        plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.bar(x_pos, importances, align="center")
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        show_or_save(export_path)


__all__ = [
    "Interpreter",
]
