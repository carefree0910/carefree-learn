from typing import Any
from typing import Optional
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type

from .core import HuggingFaceModel
from ....types import texts_type
from ....schema import TrainerState
from ....constants import PREDICTIONS_KEY

try:
    from transformers import MarianMTModel
    from transformers import MarianTokenizer
except:
    MarianMTModel = MarianTokenizer = None


@HuggingFaceModel.register("opus-base")
class OPUSBase(HuggingFaceModel):
    model_base = MarianMTModel
    tokenizer_base = MarianTokenizer
    forward_fn_name = "generate"

    def __init__(self, src: str, tgt: str) -> None:
        if MarianMTModel is None:
            raise ValueError("`transformers` is needed for `OPUSBase`")
        super().__init__(f"Helsinki-NLP/opus-mt-{src}-{tgt}")

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        generated_ids = super().model_forward(batch)
        return {PREDICTIONS_KEY: generated_ids}

    def inference(
        self,
        texts: texts_type,
        use_tqdm: bool = True,
        **kwargs: Any,
    ) -> np_dict_type:
        return super().inference(texts, use_tqdm, stack_outputs=False, **kwargs)


@HuggingFaceModel.register("opus-zh-en")
class OPUS_ZH_EN(OPUSBase):
    def __init__(self) -> None:
        super().__init__("zh", "en")


__all__ = [
    "OPUSBase",
    "OPUS_ZH_EN",
]
