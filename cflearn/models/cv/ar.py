from typing import Any
from typing import Tuple
from typing import Optional
from cftool.types import tensor_dict_type

from ..common import build_loss
from ..common import CommonDLModel
from ...schema import DLConfig
from ...schema import TrainerState
from ...modules import build_auto_regressor
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY


@CommonDLModel.register("ar")
class AutoRegressorModel(CommonDLModel):
    def build(self, config: DLConfig) -> None:
        if config.loss_name is None:
            raise ValueError("`loss_name` should be specified for `AutoRegressorModel`")
        self.m = build_auto_regressor(config.module_name, config=config.module_config)
        self.loss = build_loss(config.loss_name, config=config.loss_config)

    def get_forward_args(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> Tuple[Any, ...]:
        return batch[INPUT_KEY], batch.get(LABEL_KEY)


__all__ = [
    "AutoRegressorModel",
]
