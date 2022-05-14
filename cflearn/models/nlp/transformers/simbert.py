from typing import Any
from typing import Optional

from .core import HuggingFaceModel
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import LATENT_KEY
from ....misc.toolkit import l2_normalize


@HuggingFaceModel.register("simbert")
class SimBERT(HuggingFaceModel):
    def __init__(self) -> None:
        super().__init__("peterchou/simbert-chinese-base")

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        results = super().forward(batch_idx, batch, state, **kwargs)
        results[LATENT_KEY] = l2_normalize(results["pooler_output"])
        return results


__all__ = [
    "SimBERT",
]