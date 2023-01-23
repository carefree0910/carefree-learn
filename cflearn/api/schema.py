import numpy as np

from abc import abstractmethod
from typing import List
from typing import NamedTuple


class ImageFolderLatentResponse(NamedTuple):
    latent: np.ndarray
    img_paths: List[str]


class IImageExtractor:
    @abstractmethod
    def get_folder_latent(
        self,
        src_folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        use_tqdm: bool = True,
    ) -> ImageFolderLatentResponse:
        pass


__all__ = [
    "IImageExtractor",
]
