import numpy as np

from torch import Tensor
from typing import List
from typing import Tuple

from .utils import predict_folder
from ..pipeline import SimplePipeline
from ....types import texts_type
from ....protocol import InferenceProtocol
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....misc.toolkit import to_torch
from ....data.interface import TensorData
from ....models.nlp.tokenizers import TokenizerProtocol


class CLIPTextExtractor:
    def __init__(self, m: SimplePipeline):
        self.m = m
        clip = m.model
        self.tokenizer = TokenizerProtocol.make("clip", {})
        clip.forward = lambda _, batch, *args, **kwargs: {  # type: ignore
            LATENT_KEY: clip.encode_text(batch[INPUT_KEY]),
        }

    def get_texts_latent(
        self,
        texts: texts_type,
        *,
        batch_size: int = 64,
        use_tqdm: bool = True,
    ) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]
        text_arrays = [self.tokenizer.tokenize(t) for t in texts]
        texts_tensor = to_torch(np.vstack(text_arrays))
        data = TensorData(texts_tensor, batch_size=batch_size)
        data.prepare(None)
        loader = data.initialize()[0]
        inference = InferenceProtocol(model=self.m.model)
        outputs = inference.get_outputs(loader, use_tqdm=use_tqdm)
        return outputs.forward_results


class CLIPImageExtractor:
    def __init__(self, m: SimplePipeline):
        self.m = m
        clip = m.model
        self.img_size = clip.img_size
        self.transform = clip.get_transform()
        clip.forward = lambda _, batch, *args, **kwargs: {  # type: ignore
            LATENT_KEY: clip.encode_image(batch[INPUT_KEY]),
        }

    def get_folder_latent(
        self,
        src_folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        use_tqdm: bool = True,
    ) -> Tuple[Tensor, List[str]]:
        results = predict_folder(
            self.m,
            src_folder,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=self.transform,
            use_tqdm=use_tqdm,
        )
        latent = to_torch(results.outputs[LATENT_KEY])
        return latent, results.img_paths


__all__ = [
    "CLIPTextExtractor",
    "CLIPImageExtractor",
]
