import torch

import numpy as np
import torch.nn.functional as F

from PIL import Image
from torch import Tensor
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from cftool.array import save_images

from .common import APIMixin
from ....pipeline import DLPipeline
from ....zoo.core import DLZoo
from ....misc.toolkit import eval_context

try:
    from cfcv.misc.toolkit import to_rgb
except:
    to_rgb = None


def default_preprocess(image: Image.Image) -> np.ndarray:
    return np.array(image).astype(np.float32) / 255.0


class TranslatorAPI(APIMixin):
    def sr(
        self,
        image: Union[str, Image.Image],
        export_path: Optional[str] = None,
        *,
        clip_range: Optional[Tuple[int, int]] = (0, 1),
        preprocess_fn: Optional[Callable[[Image.Image], np.ndarray]] = None,
    ) -> Tensor:
        if isinstance(image, str):
            image = Image.open(image)
        # handle alpha
        alpha = None
        if image.mode == "RGBA":
            if to_rgb is None:
                raise ValueError("`carefree-cv` is needed for `TranslatorAPI`")
            alpha = image.split()[3]
            image = to_rgb(image)
        # inference
        array = (preprocess_fn or default_preprocess)(image)
        tensor = torch.from_numpy(array)[None].permute(0, 3, 1, 2)
        tensor = tensor.contiguous().to(self.device)
        with eval_context(self.m):
            output = self.m(tensor).cpu()
        if clip_range is not None:
            output = torch.clip(output, *clip_range)
        # handle alpha
        if alpha is not None:
            alpha_tensor = torch.from_numpy(np.array(alpha).astype(np.float32) / 255.0)
            alpha_tensor = alpha_tensor[None, None]
            with torch.no_grad():
                alpha_tensor = F.interpolate(
                    alpha_tensor,
                    output.shape[-2:],
                    mode="nearest",
                )
            output = torch.cat([output, alpha_tensor], dim=1)
        # export
        if export_path is not None:
            save_images(output, export_path)
        return output

    @classmethod
    def from_esr(
        cls,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
    ) -> "TranslatorAPI":
        return cls.from_pipeline(esr(), device, use_amp=use_amp)


def esr(pretrained: bool = True) -> DLPipeline:
    return DLZoo.load_pipeline("sr/esr", pretrained=pretrained)


__all__ = [
    "TranslatorAPI",
]
