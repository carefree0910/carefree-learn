import math
import torch

from abc import abstractmethod
from abc import ABCMeta
from torch import nn
from torch import Tensor
from typing import Any
from typing import Tuple
from typing import Optional
from cftool.types import tensor_dict_type

from .constants import TEXT_KEY
from ...schema import TrainerState
from ...schema import IDLModel
from ...constants import INPUT_KEY
from ...constants import PREDICTIONS_KEY


class IPerceptor(IDLModel, metaclass=ABCMeta):
    def __init__(self, img_size: int, context_length: int):
        super().__init__()
        self.img_size = img_size
        self.context_length = context_length
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @abstractmethod
    def encode_image(self, image: Tensor) -> Tensor:
        pass

    @abstractmethod
    def encode_text(self, text: Tensor) -> Tensor:
        pass

    def get_forward_args(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> Tuple[Any, ...]:
        return batch[INPUT_KEY], batch[TEXT_KEY]

    def forward(self, image: Tensor, text: Tensor) -> tensor_dict_type:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return {PREDICTIONS_KEY: logits_per_image}


__all__ = [
    "IPerceptor",
]
