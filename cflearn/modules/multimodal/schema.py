import math
import torch

from abc import abstractmethod
from abc import ABCMeta
from torch import nn
from torch import Tensor


class IPerceptor(nn.Module, metaclass=ABCMeta):
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

    def forward(self, image: Tensor, text: Tensor) -> Tensor:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image


__all__ = [
    "IPerceptor",
]
