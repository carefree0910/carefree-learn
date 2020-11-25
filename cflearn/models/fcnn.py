import numpy as np

from typing import Any
from typing import Optional

from .base import ModelBase
from ..types import tensor_dict_type


@ModelBase.register("fcnn")
@ModelBase.register_pipe("fcnn")
class FCNN(ModelBase):
    pass


@ModelBase.register("q_fcnn")
@ModelBase.register_pipe("fcnn")
class QuantileFCNN(ModelBase):
    def _init_config(self) -> None:
        super()._init_config()
        self.fetch_q = True
        self.fetch_cdf = False
        quantiles = self.config.setdefault("quantiles", [10, 30, 50, 70, 90])
        quantiles = list(map(int, map(round, quantiles)))
        try:
            self.median_idx = quantiles.index(50)
        except ValueError:
            raise ValueError("median (50) should be included in `quantiles`")
        self.quantiles = quantiles
        self.config["loss"] = "quantile"
        self.config["loss_config"] = {"q": [q / 100 for q in quantiles]}
        fcnn = self.pipe_configs.setdefault("fcnn", {})
        head_config = fcnn.setdefault("head", {})
        head_config["out_dim"] = len(quantiles)

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        batch_step: int = 0,
        **kwargs: Any,
    ) -> tensor_dict_type:
        results = super().forward(
            batch,
            batch_indices,
            loader_name,
            batch_step,
            **kwargs,
        )
        predictions = results["predictions"]
        median = predictions[..., self.median_idx : self.median_idx + 1]
        results = {"predictions": median, "quantiles": predictions}
        q = kwargs.get("q")
        if q is not None:
            q = int(round(100 * q))
            try:
                idx = self.quantiles.index(q)
            except ValueError:
                msg = f"quantile '{q}' is not included in the preset quantiles"
                raise ValueError(msg)
            results["quantiles"] = predictions[..., idx : idx + 1]
        return results


__all__ = ["FCNN"]
