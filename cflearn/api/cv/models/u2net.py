import os
import torch

import numpy as np

from PIL import Image
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from skimage import io
from skimage.filters import gaussian
from skimage.filters import unsharp_mask
from onnxruntime import InferenceSession
from torchvision.transforms import Compose

from ..data import RescaleT
from ..data import ToNormalizedArray
from ....trainer import DeviceInfo
from ....constants import INPUT_KEY
from ....constants import WARNING_PREFIX
from ....misc.toolkit import to_numpy
from ....misc.toolkit import to_torch
from ....misc.toolkit import to_standard
from ....misc.toolkit import eval_context
from ....misc.toolkit import naive_cutout
from ....misc.toolkit import min_max_normalize
from ....misc.toolkit import alpha_matting_cutout
from ....models.cv import U2Net


def cutout(
    img: np.ndarray,
    alpha: np.ndarray,
    smooth: int = 4,
    tight: float = 0.9,
    alpha_matting_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    alpha_im = Image.fromarray(min_max_normalize(alpha))
    alpha_im = alpha_im.resize((img.shape[1], img.shape[0]), Image.NEAREST)
    alpha = gaussian(np.array(alpha_im), smooth)
    alpha = unsharp_mask(alpha, smooth, smooth * tight)
    alpha = min_max_normalize(alpha)
    if alpha_matting_config is None:
        rgba = naive_cutout(img, alpha)
    else:
        try:
            rgba = alpha_matting_cutout(img, alpha, **alpha_matting_config)
        except Exception as err:
            print(
                f"{WARNING_PREFIX}alpha_matting failed ({err}), "
                f"naive cutting will be used"
            )
            rgba = naive_cutout(img, alpha)
    return alpha, rgba


def export(rgba: np.ndarray, tgt_path: Optional[str]) -> None:
    if tgt_path is not None:
        folder = os.path.split(tgt_path)[0]
        os.makedirs(folder, exist_ok=True)
        io.imsave(tgt_path, rgba)


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

    def _generate(
        self,
        src: np.ndarray,
        smooth: int,
        tight: float,
        alpha_matting_config: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        transformed = self.transform({INPUT_KEY: src})[INPUT_KEY]
        tensor = to_torch(transformed).to(self.device)[None, ...]
        with eval_context(self.model):
            tensor = torch.sigmoid(self.model.generate_from(tensor)[0][0])
        alpha = to_numpy(tensor)
        return cutout(src, alpha, smooth, tight, alpha_matting_config)

    def generate_alpha(
        self,
        src_path: str,
        tgt_path: Optional[str] = None,
        *,
        smooth: int = 16,
        tight: float = 0.5,
        alpha_matting_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        img = io.imread(src_path)
        img = img.astype(np.float32) / 255.0
        alpha, rgba = self._generate(img, smooth, tight, alpha_matting_config)
        export(rgba, tgt_path)
        return alpha, rgba


class U2NetAPIWithONNX:
    def __init__(
        self,
        onnx_path: str,
        rescale_size: int = 320,
    ):
        self.ort_session = InferenceSession(onnx_path)
        self.transform = Compose([RescaleT(rescale_size), ToNormalizedArray()])

    def _generate(
        self,
        src: np.ndarray,
        smooth: int,
        tight: float,
        alpha_matting_config: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        transformed = self.transform({INPUT_KEY: src})[INPUT_KEY][None, ...]
        ort_inputs = {
            node.name: to_standard(transformed)
            for node in self.ort_session.get_inputs()
        }
        logits = self.ort_session.run(None, ort_inputs)[0][0][0]
        logits = np.clip(logits, -50.0, 50.0)
        alpha = 1.0 / (1.0 + np.exp(-logits))
        return cutout(src, alpha, smooth, tight, alpha_matting_config)

    def generate_alpha(
        self,
        src_path: str,
        tgt_path: Optional[str] = None,
        *,
        smooth: int = 16,
        tight: float = 0.5,
        alpha_matting_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        img = io.imread(src_path)
        img = img.astype(np.float32) / 255.0
        alpha, rgba = self._generate(img, smooth, tight, alpha_matting_config)
        export(rgba, tgt_path)
        return alpha, rgba


__all__ = [
    "U2NetAPI",
    "U2NetAPIWithONNX",
]
