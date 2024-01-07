import torch

import numpy as np

from PIL import Image
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from cftool.array import to_torch

from ..common import IAPI
from ...data import collect_images
from ...data import ArrayData
from ...schema import texts_type
from ...schema import device_type
from ...schema import DataConfig
from ...models import CommonDLModel
from ...modules import CLIP
from ...modules import ITokenizer
from ...toolkit import get_device
from ...inference import DLInference
from ...constants import INPUT_KEY
from ...constants import PREDICTIONS_KEY


class CLIPExtractor(IAPI):
    m: CLIP

    def __init__(
        self,
        m: CLIP,
        device: device_type = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        force_not_lazy: bool = False,
        tokenizer: Optional[str] = None,
        pad_to_max: bool = True,
    ):
        super().__init__(
            m,
            device,
            use_amp=use_amp,
            use_half=use_half,
            force_not_lazy=force_not_lazy,
        )
        self.img_size = m.img_size
        self.transform = m.get_transform()
        if tokenizer is None:
            if m.context_length == 512:
                tokenizer = "clip.chinese"
            else:
                tokenizer = "clip"
        self.tokenizer = ITokenizer.make(tokenizer, dict(pad_to_max=pad_to_max))

    @property
    def model(self) -> CommonDLModel:
        model = CommonDLModel()
        model.m = self.m
        model.loss = None  # type: ignore
        return model

    @property
    def text_forward_fn(self) -> Callable:
        return lambda net: self.m.encode_text(net)

    @property
    def image_forward_fn(self) -> Callable:
        return lambda net: self.m.encode_image(net)

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
        data_config = DataConfig.inference_with(batch_size)
        data: ArrayData = ArrayData.init(data_config).fit(texts_tensor)
        loader = data.get_loaders()[0]
        original_forward = self.m.forward
        self.m.forward = self.text_forward_fn  # type: ignore
        inference = DLInference(model=self.model)
        outputs = inference.get_outputs(loader, use_tqdm=use_tqdm)
        self.m.forward = original_forward  # type: ignore
        return outputs.forward_results[PREDICTIONS_KEY]

    def get_image_latent(
        self,
        images: Union[Image.Image, List[Image.Image]],
        *,
        batch_size: int = 64,
        use_tqdm: bool = True,
    ) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]
        images_tensor = torch.stack([self.transform(image) for image in images])
        images_tensor = images_tensor.to(get_device(self.m))
        data_config = DataConfig.inference_with(batch_size)
        data: ArrayData = ArrayData.init(data_config).fit(images_tensor)
        loader = data.get_loaders()[0]
        original_forward = self.m.forward
        self.m.forward = self.image_forward_fn  # type: ignore
        inference = DLInference(model=self.model)
        outputs = inference.get_outputs(loader, use_tqdm=use_tqdm)
        self.m.forward = original_forward  # type: ignore
        return outputs.forward_results[PREDICTIONS_KEY]

    def get_paths_latent(
        self,
        image_paths: List[str],
        *,
        batch_size: int = 64,
        use_tqdm: bool = True,
    ) -> np.ndarray:
        images = [Image.open(image_path) for image_path in image_paths]
        return self.get_image_latent(images, batch_size=batch_size, use_tqdm=use_tqdm)

    def get_folder_latent(
        self,
        image_folder: str,
        *,
        batch_size: int = 64,
        use_tqdm: bool = True,
    ) -> np.ndarray:
        paths = collect_images(image_folder).all_img_paths
        return self.get_paths_latent(paths, batch_size=batch_size, use_tqdm=use_tqdm)

    def to_text_onnx(
        self,
        export_folder: str,
        *,
        num_samples: Optional[int] = None,
        onnx_file: str = "text.onnx",
    ) -> None:
        inp = to_torch(self.tokenizer.tokenize("Test."))
        self.model.to_onnx(
            export_folder,
            {INPUT_KEY: inp},
            onnx_file=onnx_file,
            forward_fn=self.text_forward_fn,
            output_names=[PREDICTIONS_KEY],
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
        self.model.to_onnx(
            export_folder,
            {INPUT_KEY: inp},
            onnx_file=onnx_file,
            forward_fn=self.image_forward_fn,
            output_names=[PREDICTIONS_KEY],
            num_samples=num_samples,
        )


__all__ = [
    "CLIPExtractor",
]
