from torch import Tensor
from typing import List
from typing import Tuple

from .utils import predict_folder
from ..pipeline import SimplePipeline
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....misc.toolkit import to_torch


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
    "CLIPImageExtractor",
]
