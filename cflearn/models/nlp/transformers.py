from typing import Any
from typing import Optional
from transformers import AutoModel
from transformers import AutoTokenizer

from ..bases import ModelProtocol
from ...types import texts_type
from ...types import tensor_dict_type
from ...protocol import TrainerState


@ModelProtocol.register("hugging_face")
class HuggingFaceModel(ModelProtocol):
    def __init__(self, model: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return self.model(**batch)

    def inference(self, texts: texts_type) -> tensor_dict_type:
        return self.forward(0, self.tokenizer(texts))


__all__ = [
    "HuggingFaceModel",
]
