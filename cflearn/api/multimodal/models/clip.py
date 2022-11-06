import torch

import numpy as np

from PIL import Image
from typing import List
from typing import Callable
from typing import Optional
from cftool.array import to_torch

from ...protocol import IImageExtractor
from ...protocol import ImageFolderLatentResponse
from ...cv.utils import predict_paths
from ...cv.utils import predict_folder
from ...cv.pipeline import CVPipeline
from ....data import TensorData
from ....types import texts_type
from ....protocol import IInference
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....misc.toolkit import eval_context
from ....models.nlp.tokenizers import ITokenizer
from ....models.multimodal.clip import CLIP


class CLIPExtractor(IImageExtractor):
    clip: CLIP

    def __init__(
        self,
        m: CVPipeline,
        *,
        tokenizer: Optional[str] = None,
        pad_to_max: bool = True,
    ):
        self.m = m
        clip = m.model
        self.clip = clip
        self.img_size = clip.img_size
        self.transform = clip.get_transform()
        if tokenizer is None:
            if self.clip.context_length == 512:
                tokenizer = "clip.chinese"
            else:
                tokenizer = "clip"
        self.tokenizer = ITokenizer.make(tokenizer, dict(pad_to_max=pad_to_max))

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
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts_tensor = to_torch(self.tokenizer.tokenize(texts))
        loader = TensorData(texts_tensor, batch_size=batch_size).get_loaders()[0]
        original_forward = self.clip.forward
        self.clip.forward = lambda _, batch, *s, **kws: self.text_forward_fn(batch)  # type: ignore
        inference = IInference(model=self.clip)
        outputs = inference.get_outputs(loader, use_tqdm=use_tqdm)
        self.clip.forward = original_forward  # type: ignore
        return outputs.forward_results[LATENT_KEY]

    def get_image_latent(self, image: Image.Image) -> np.ndarray:
        inp = self.transform(image)[None].to(self.clip.device)
        with eval_context(self.clip):
            return self.clip.encode_image(inp).cpu().numpy()

    def get_paths_latent(
        self,
        image_paths: List[str],
        *,
        batch_size: int = 64,
        num_workers: int = 0,
        use_tqdm: bool = True,
    ) -> np.ndarray:
        original_forward = self.clip.forward
        self.clip.forward = lambda _, batch, *s, **kws: self.image_forward_fn(batch)  # type: ignore
        results = predict_paths(
            self.m,
            image_paths,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=self.transform,
            use_tqdm=use_tqdm,
        )
        self.clip.forward = original_forward  # type: ignore
        return results.outputs[LATENT_KEY]

    def get_folder_latent(
        self,
        src_folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        use_tqdm: bool = True,
    ) -> ImageFolderLatentResponse:
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
        return ImageFolderLatentResponse(results.outputs[LATENT_KEY], results.img_paths)

    def to_text_onnx(
        self,
        export_folder: str,
        *,
        num_samples: Optional[int] = None,
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
        num_samples: Optional[int] = None,
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
