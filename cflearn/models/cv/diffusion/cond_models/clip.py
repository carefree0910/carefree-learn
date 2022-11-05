import re
import torch

from torch import Tensor
from typing import Any
from typing import List

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
    res = []
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

    def get_line(self, text: str) -> Tensor:
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
