import torch

from typing import List
from typing import Optional
from typing import NamedTuple

from ...common import IAPI
from ....schema import device_type

try:
    from transformers import GPT2Tokenizer
    from transformers import GPT2LMHeadModel
except:
    GPT2Tokenizer = GPT2LMHeadModel = None


class PromptConfig(NamedTuple):
    temperature: float = 0.9
    top_k: int = 8
    max_length: int = 76
    repitition_penalty: float = 1.2
    num_return_sequences: int = 1
    comma_mode: bool = False


class PromptEnhanceAPI(IAPI):
    def __init__(
        self,
        device: device_type = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        force_not_lazy: bool = False
    ):
        if GPT2Tokenizer is None or GPT2LMHeadModel is None:
            raise ValueError("`trainsformers` is required for `PromptEnhanceAPI`")
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        version = "FredZhang7/distilgpt2-stable-diffusion-v2"
        super().__init__(
            GPT2LMHeadModel.from_pretrained(version),
            device,
            use_amp=use_amp,
            use_half=use_half,
            force_not_lazy=force_not_lazy,
        )

    @torch.no_grad()
    def enhance(self, prompt: str, config: Optional[PromptConfig] = None) -> List[str]:
        if config is None:
            config = PromptConfig()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        kw = dict(
            do_sample=True,
            temperature=config.temperature,
            top_k=config.top_k,
            max_length=config.max_length,
            num_return_sequences=config.num_return_sequences,
            repetition_penalty=config.repitition_penalty,
            early_stopping=True,
        )
        if not config.comma_mode:
            kw.update(dict(penalty_alpha=0.6, no_repeat_ngram_size=1))
        outputs = self.m.generate(input_ids, **kw)
        return [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]


__all__ = [
    "PromptConfig",
    "PromptEnhanceAPI",
]
