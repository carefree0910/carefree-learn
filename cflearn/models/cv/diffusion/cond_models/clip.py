import numpy as np

from torch import Tensor
from typing import List
from typing import Callable
from cftool.array import to_torch

from .protocol import IConditionModel
from ....nlp.tokenizers import ITokenizer
from ....multimodal.clip import CLIP


class ICLIPTokenizer:
    encoder: dict
    tokenize: Callable


@IConditionModel.register("multimodal/clip")
@IConditionModel.register("multimodal/clip.large")
@IConditionModel.register("multimodal/clip.chinese")
class CLIPTextConditionModel(IConditionModel):
    m: CLIP
    tokenizer: ICLIPTokenizer

    def __init__(self, m: CLIP):
        super().__init__(m)
        tokenizer = "clip.chinese" if m.context_length == 512 else "clip"
        self.tokenizer = ITokenizer.make(tokenizer, {})

    def get_text_indices(self, texts: List[str]) -> Tensor:
        text_arrays = [self.tokenizer.tokenize(t) for t in texts]
        text = to_torch(np.vstack(text_arrays)).to(self.m.device)
        text[text == 0] = len(self.tokenizer.encoder) - 1
        return text

    def forward(self, cond: List[str]) -> Tensor:
        text = self.get_text_indices(cond)
        return self.m.encode_text(text, apply_pooling=False, determinate=False)


__all__ = [
    "CLIPTextConditionModel",
]
