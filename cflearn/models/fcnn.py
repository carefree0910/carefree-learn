import numpy as np

from typing import Any
from typing import Optional

from .base import ModelBase
from ..types import tensor_dict_type
from ..protocol import TrainerState


@ModelBase.register("fcnn")
@ModelBase.register_pipe("fcnn")
class FCNN(ModelBase):
    pass


@ModelBase.register("q_fcnn")
@ModelBase.register_pipe("fcnn")
class QuantileFCNN(ModelBase):
    def _init_config(self) -> None:
        super()._init_config()
        quantiles = self.config.setdefault("quantiles", [10, 30, 50, 70, 90])
        quantiles = list(map(int, map(round, quantiles)))
        try:
            self.median_idx = quantiles.index(50)
        except ValueError:
            raise ValueError("median (50) should be included in `quantiles`")
        self.quantiles = quantiles
        self.config["loss"] = "quantile"
        self.config["loss_config"] = {"q": [q / 100 for q in quantiles]}
        head_config = self.get_pipe_config("fcnn", "head")
        head_config["out_dim"] = len(quantiles)

    def forward(
        self,
        batch: tensor_dict_type,
        batch_idx: Optional[int] = None,
        state: Optional[TrainerState] = None,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        results = super().forward(
            batch,
            batch_idx,
            state,
            batch_indices,
            loader_name,
            **kwargs,
        )
        predictions = results["predictions"]
        median = predictions[..., [self.median_idx]]
        results = {"predictions": median, "quantiles": predictions}
        q = kwargs.get("q")
        if q is not None:
            fn = lambda q_: int(round(100 * q_))
            if not isinstance(q, (list, tuple)):
                q_list = [fn(q)]
            else:
                q_list = list(map(fn, q))
            indices = []
            for q in q_list:
                try:
                    indices.append(self.quantiles.index(q))
                except ValueError:
                    msg = f"quantile '{q}' is not included in the preset quantiles"
                    raise ValueError(msg)
            results["quantiles"] = predictions[..., indices]
        return results


__all__ = ["FCNN", "QuantileFCNN"]
