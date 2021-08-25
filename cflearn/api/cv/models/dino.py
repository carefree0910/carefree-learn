import torch

from PIL import Image
from tqdm import tqdm
from torch import Tensor
from typing import List
from typing import Tuple

from ..data import SSLTestTransform
from ..data import InferenceImageFolderDataset
from ....constants import LATENT_KEY
from ....misc.toolkit import to_device
from ....misc.toolkit import eval_context
from ....models.cv import DINO


class DINOPredictor:
    def __init__(self, dino: DINO):
        self.dino = dino
        img_size = dino.student.backbone.img_size
        self.transform = SSLTestTransform(img_size)

    @property
    def device(self) -> torch.device:
        return self.dino.device

    def get_latent(self, src_path: str) -> Tensor:
        src = Image.open(src_path).convert("RGB")
        net = self.transform(src)[None, ...].to(self.device)
        with eval_context(self.dino):
            return self.dino.get_latent(net)

    def get_folder_latent(
        self,
        src_folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        use_tqdm: bool = True,
    ) -> Tuple[Tensor, List[str]]:
        dataset = InferenceImageFolderDataset(src_folder, self.transform)
        loader = dataset.make_loader(batch_size, num_workers)
        outputs = []
        iterator = enumerate(loader)
        if use_tqdm:
            iterator = tqdm(iterator, total=len(loader))
        with eval_context(self.dino):
            for i, batch in iterator:
                batch = to_device(batch, self.device)
                outputs.append(self.dino(i, batch)[LATENT_KEY].cpu())
        return torch.cat(outputs, dim=0), dataset.img_paths


__all__ = [
    "DINOPredictor",
]
