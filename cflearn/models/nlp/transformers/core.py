from typing import Any
from typing import Optional
from transformers import AutoModel
from transformers import AutoTokenizer

from ...bases import ModelProtocol
from ....types import texts_type
from ....types import np_dict_type
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....protocol import InferenceProtocol
from ....data.interface import TensorDictData
from ....misc.toolkit import check_requires
from ....misc.toolkit import shallow_copy_dict


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
        fn = self.model.forward
        batch = shallow_copy_dict(batch)
        pop_keys = [k for k in batch.keys() if not check_requires(fn, k, False)]
        for k in pop_keys:
            batch.pop(k)
        return self.model(**batch)

    def inference(self, texts: texts_type, use_tqdm: bool = True) -> np_dict_type:
        x = self.tokenizer(texts, padding=True, return_tensors="pt")
        data = TensorDictData(x)
        data.prepare(None)
        loader = data.initialize()[0]
        inference = InferenceProtocol(model=self)
        return inference.get_outputs(loader, use_tqdm=use_tqdm).forward_results


__all__ = [
    "HuggingFaceModel",
]
