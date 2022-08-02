import torch

import numpy as np

from torch import Tensor
from typing import List
from typing import Tuple
from typing import Callable
from typing import Optional
from cftool.array import to_torch

from ...cv.models.utils import predict_folder
from ...cv.pipeline import SimplePipeline
from ....types import texts_type
from ....protocol import InferenceProtocol
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....data.interface import TensorData
from ....models.nlp.tokenizers import TokenizerProtocol
from ....models.multimodal.clip import CLIP


class CLIPExtractor:
    clip: CLIP

    def __init__(self, m: SimplePipeline):
        self.m = m
        clip = m.model
        self.clip = clip
        self.img_size = clip.img_size
        self.transform = clip.get_transform()
        self.tokenizer = TokenizerProtocol.make("clip", {})

    @property
    def text_forward_fn(self) -> Callable:
        return lambda batch: {LATENT_KEY: self.clip.encode_text(batch[INPUT_KEY])}

    @property
    def image_forward_fn(self) -> Callable:
        return lambda batch: {LATENT_KEY: self.clip.encode_image(batch[INPUT_KEY])}

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
        loader = TensorData(texts_tensor, batch_size=batch_size).get_loaders()[0]
        original_forward = self.clip.forward
        self.clip.forward = lambda _, batch, *s, **kws: self.text_forward_fn(batch)  # type: ignore
        inference = InferenceProtocol(model=self.clip)
        outputs = inference.get_outputs(loader, use_tqdm=use_tqdm)
        self.clip.forward = original_forward  # type: ignore
        return outputs.forward_results[LATENT_KEY]

    def get_folder_latent(
        self,
        src_folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        use_tqdm: bool = True,
    ) -> Tuple[Tensor, List[str]]:
        original_forward = self.clip.forward
        self.clip.forward = lambda _, batch, *s, **kws: self.image_forward_fn(batch)  # type: ignore
        results = predict_folder(
            self.m,
            src_folder,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=self.transform,
            use_tqdm=use_tqdm,
        )
        self.clip.forward = original_forward  # type: ignore
        latent = to_torch(results.outputs[LATENT_KEY])
        return latent, results.img_paths

    def to_text_onnx(
        self,
        export_folder: str,
        *,
        num_samples: Optional[int] = 1,
        onnx_file: str = "text.onnx",
    ) -> None:
        inp = to_torch(self.tokenizer.tokenize("Test."))
        self.clip.to_onnx(
            export_folder,
            {INPUT_KEY: inp},
            onnx_file=onnx_file,
            forward_fn=self.text_forward_fn,
            output_names=[LATENT_KEY],
            num_samples=num_samples,
        )

    def to_image_onnx(
        self,
        export_folder: str,
        *,
        num_samples: Optional[int] = 1,
        onnx_file: str = "image.onnx",
    ) -> None:
        inp = torch.randn(1, 3, self.img_size, self.img_size)
        self.clip.to_onnx(
            export_folder,
            {INPUT_KEY: inp},
            onnx_file=onnx_file,
            forward_fn=self.image_forward_fn,
            output_names=[LATENT_KEY],
            num_samples=num_samples,
        )


__all__ = [
    "CLIPExtractor",
]
