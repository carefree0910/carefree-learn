import re
import dill
import uuid
import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple

from .protocol import IConditionModel
from ....nlp.tokenizers import ITokenizer
from ....nlp.tokenizers import ICLIPTokenizer
from ....multimodal.clip import CLIP


weight_pattern = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_weights(text: str) -> List[List[Any]]:
    res: List[List[Any]] = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position: int, multiplier: float) -> None:
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in weight_pattern.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    i = 0
    while i + 1 < len(res):
        if res[i][1] != res[i + 1][1]:
            i += 1
        else:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)

    return res


class CustomToken(NamedTuple):
    name: str
    uuid: str
    token_id: int
    embedding: Tensor


class inject_embeddings:
    def __init__(self, cond_model: "CLIPTextConditionModel") -> None:
        self.m = cond_model
        self.original_embeddings = cond_model.embeddings
        self.changed = False

    def __enter__(self) -> None:
        sorted_keys = sorted(self.m.customized)
        if not sorted_keys:
            return
        anchor = self.original_embeddings
        self.changed = True
        new_inject = [self.m.customized[k].embedding.to(anchor) for k in sorted_keys]
        start = len(anchor)
        for i, k in enumerate(sorted_keys):
            if start + i != k:
                msg = "injected embeddings not contiguous to the original embeddings"
                raise ValueError(msg)
        concat = torch.cat([anchor] + new_inject, dim=0)
        self._inject(concat)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.changed:
            return
        self._inject(self.original_embeddings)

    def _inject(self, embeddings: Tensor) -> None:
        if self.m.m.token_embedding is None:
            raise ValueError("`token_embedding` is None")
        self.m.m.token_embedding.weight.data = embeddings


@IConditionModel.register("multimodal/clip")
@IConditionModel.register("multimodal/clip.large")
@IConditionModel.register("multimodal/clip.chinese")
class CLIPTextConditionModel(IConditionModel):
    m: CLIP
    tokenizer: ICLIPTokenizer

    def __init__(self, m: CLIP):
        super().__init__(m)
        self.context_length = m.context_length
        tokenizer = "clip.chinese" if m.context_length == 512 else "clip"
        self.tokenizer = ITokenizer.make(tokenizer, dict(pad_to_max=True))
        self._dumped_tokenizer = dill.dumps(self.tokenizer.tokenizer)
        self.dictionary: Dict[str, str] = {}
        self.customized: Dict[int, CustomToken] = {}

    @property
    def embeddings(self) -> Tensor:
        if self.m.token_embedding is None:
            raise ValueError("`token_embedding` is None")
        return self.m.token_embedding.weight.data

    def register_custom(self, embeddings: Dict[str, List[float]]) -> None:
        existing = self.embeddings
        dtype = existing.dtype
        device = existing.device
        for name, embedding in embeddings.items():
            tensor = torch.asarray(embedding, dtype=dtype, device=device).view(1, -1)
            for token_id, token in list(self.customized.items()):
                if name == token.name:
                    new_token = CustomToken(token.name, token.uuid, token_id, tensor)
                    self.customized[token_id] = new_token
                    return
            tag = uuid.uuid4().hex
            self.tokenizer.tokenizer.add_tokens(tag)
            token_id = self.tokenizer.tokenizer.convert_tokens_to_ids(tag)
            self.dictionary[name] = tag
            self.customized[token_id] = CustomToken(name, tag, token_id, tensor)

    def clear_custom(self) -> None:
        self.dictionary = {}
        self.customized = {}
        # TODO : optimize this
        self.tokenizer.tokenizer = dill.loads(self._dumped_tokenizer)

    def get_line(self, text: str) -> Tensor:
        for k, v in self.dictionary.items():
            text = text.replace(k, v)
        parsed = parse_weights(text)
        parsed_texts = [pair[0] for pair in parsed]
        encoded = self.tokenizer.encode(
            parsed_texts,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]
        concat_ids = [self.tokenizer.bos_token_id]
        weights = [1.0]
        for ids, (_, weight) in zip(encoded, parsed):
            concat_ids += ids
            weights += [weight] * len(ids)
        # padding
        diff = self.context_length - len(concat_ids)
        if diff > 0:
            concat_ids += [self.tokenizer.eos_token_id] * diff
            weights += [1.0] * diff
        else:
            concat_ids = concat_ids[: self.context_length - 1]
            weights = weights[: self.context_length - 1]
            concat_ids.append(self.tokenizer.eos_token_id)
            weights.append(1.0)
        # encode
        to_torch = lambda l, dtype: torch.asarray(l, dtype=dtype, device=self.m.device)
        inp = to_torch([concat_ids], torch.int64)
        weights_tensor = to_torch(weights, torch.float32).view(1, -1, 1)
        with inject_embeddings(self):
            z = self.m.encode_text(inp, apply_pooling=False, determinate=False)
        original_mean = z.mean()
        z *= weights_tensor
        new_mean = z.mean()
        z *= original_mean / new_mean
        return z

    def forward(self, cond: List[str]) -> Tensor:
        lines = list(map(self.get_line, cond))
        return torch.cat(lines, dim=0)


__all__ = [
    "CLIPTextConditionModel",
]
