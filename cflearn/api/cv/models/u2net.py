import os
import torch

import numpy as np

from PIL import Image
from typing import Optional
from skimage import io
from torchvision.transforms import Compose

from ..data import RescaleT
from ..data import ToNormalizedArray
from ....trainer import DeviceInfo
from ....constants import INPUT_KEY
from ....misc.toolkit import to_numpy
from ....misc.toolkit import to_torch
from ....misc.toolkit import eval_context
from ....misc.toolkit import min_max_normalize
from ....models.cv import U2Net


def align_to(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    im = Image.fromarray(src)
    im = im.resize((tgt.shape[1], tgt.shape[0]), resample=Image.BILINEAR)
    return np.array(im)


class U2NetAPI:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        pt_path: str,
        latent_channels: int = 32,
        num_layers: int = 5,
        max_layers: int = 7,
        lite: bool = False,
        cuda: Optional[str] = None,
        rescale_size: int = 320,
    ):
        self.device = DeviceInfo(cuda, None).device
        self.model = U2Net(
            in_channels,
            out_channels,
            latent_channels=latent_channels,
            num_layers=num_layers,
            max_layers=max_layers,
            lite=lite,
        )
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(pt_path, map_location=self.device))
        self.transform = Compose([RescaleT(rescale_size), ToNormalizedArray()])

    def generate_alpha(
        self,
        src_path: str,
        tgt_path: Optional[str] = None,
    ) -> np.ndarray:
        img = io.imread(src_path)
        img = img.astype(np.float32) / 255.0
        transformed = self.transform({INPUT_KEY: img})[INPUT_KEY]
        tensor = to_torch(transformed).to(self.device)[None, ...]
        with eval_context(self.model):
            tensor = torch.sigmoid(self.model.generate_from(tensor))
        alpha = min_max_normalize(to_numpy(tensor)[0][0])
        alpha = align_to(alpha, img)
        if tgt_path is not None:
            folder = os.path.split(tgt_path)[0]
            os.makedirs(folder, exist_ok=True)
            rgba = np.concatenate([img, alpha[..., None]], axis=2)
            rgba = (rgba * 255.0).astype(np.uint8)
            io.imsave(tgt_path, rgba)
        return alpha


__all__ = ["U2NetAPI"]
