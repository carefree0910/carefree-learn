import torch

from PIL import Image
from tqdm import tqdm
from torch import Tensor
from typing import List
from typing import Tuple
from cftool.array import to_torch
from cftool.array import to_device

from ..utils import predict_folder
from ..pipeline import CVPipeline
from ...protocol import IImageExtractor
from ...protocol import ImageFolderLatentResponse
from ....data import SSLTestTransform
from ....data import InferenceImageFolderData
from ....constants import LATENT_KEY
from ....misc.toolkit import eval_context

try:
    from cfcv.misc.toolkit import to_rgb
except:
    to_rgb = None


class DINOExtractor(IImageExtractor):
    def __init__(self, m: CVPipeline, img_size: int, *, to_gray: bool = False):
        if to_rgb is None:
            raise ValueError("`carefree-cv` is needed for `DINOExtractor`")
        self.m = m
        self.dino = m.model
        self.transform = SSLTestTransform(img_size, to_gray)

    @property
    def device(self) -> torch.device:
        return self.dino.device

    def get_latent(self, src_path: str) -> Tensor:
        src = to_rgb(Image.open(src_path))
        net = to_torch(self.transform(src)[None, ...]).to(self.device)
        with eval_context(self.dino):
            return self.dino.get_latent(net)

    def get_logits(self, src_path: str) -> Tensor:
        src = to_rgb(Image.open(src_path))
        net = self.transform(src)[None, ...].to(self.device)
        with eval_context(self.dino):
            return self.dino.get_logits(net)

    def get_folder_latent(
        self,
        src_folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        use_tqdm: bool = True,
    ) -> ImageFolderLatentResponse:
        results = predict_folder(
            self.m,
            src_folder,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=self.transform,
            use_tqdm=use_tqdm,
        )
        return ImageFolderLatentResponse(results.outputs[LATENT_KEY], results.img_paths)

    def get_folder_logits(
        self,
        src_folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        use_tqdm: bool = True,
    ) -> Tuple[Tensor, List[str]]:
        data = InferenceImageFolderData(
            src_folder,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=self.transform,
        )
        outputs = []
        iterator = data.initialize()[0]
        if use_tqdm:
            iterator = tqdm(iterator, total=len(iterator))
        with eval_context(self.dino):
            for i, batch in enumerate(iterator):
                batch = to_device(batch, self.device)
                outputs.append(self.dino.student(i, batch).cpu())
        return torch.cat(outputs, dim=0), data.dataset.img_paths


__all__ = [
    "DINOExtractor",
]
