import torch

from PIL import Image
from tqdm import tqdm
from torch import Tensor
from typing import List
from typing import Tuple

from ..data import SSLTestTransform
from ..data import InferenceImageFolderData
from ..pipeline import SimplePipeline
from ..interface import predict_folder
from ....constants import LATENT_KEY
from ....misc.toolkit import to_torch
from ....misc.toolkit import to_device
from ....misc.toolkit import eval_context


class DINOPredictor:
    def __init__(self, m: SimplePipeline, img_size: int, *, to_gray: bool = False):
        self.m = m
        self.dino = m.model
        self.transform = SSLTestTransform(img_size, to_gray)

    @property
    def device(self) -> torch.device:
        return self.dino.device

    def get_latent(self, src_path: str) -> Tensor:
        src = Image.open(src_path).convert("RGB")
        net = to_torch(self.transform(src)[None, ...]).to(self.device)
        with eval_context(self.dino):
            return self.dino.get_latent(net)

    def get_logits(self, src_path: str) -> Tensor:
        src = Image.open(src_path).convert("RGB")
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
    "DINOPredictor",
]
