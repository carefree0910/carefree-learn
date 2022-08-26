from torch import Tensor
from typing import List
from typing import Tuple


class IImageExtractor:
    def get_folder_latent(
        self,
        src_folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        use_tqdm: bool = True,
    ) -> Tuple[Tensor, List[str]]:
        pass


__all__ = [
    "IImageExtractor",
]
