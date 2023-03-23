import torch

import torch.nn.functional as F

from PIL import Image
from torch import Tensor
from typing import Tuple
from typing import Union
from typing import Optional
from cftool.cv import read_image
from cftool.cv import save_images

from ..utils import APIMixin
from ...zoo import DLZoo
from ...zoo.core import TPipeline
from ...misc.toolkit import eval_context


class TranslatorAPI(APIMixin):
    def sr(
        self,
        image: Union[str, Image.Image],
        export_path: Optional[str] = None,
        *,
        max_wh: int = 768,
        clip_range: Optional[Tuple[int, int]] = (0, 1),
    ) -> Tensor:
        res = read_image(image, max_wh, anchor=None)
        # inference
        tensor = torch.from_numpy(res.image)
        tensor = tensor.contiguous().to(self.device)
        if self.use_half:
            tensor = tensor.half()
        with eval_context(self.m):
            output = self.m(tensor).cpu().float()
        if clip_range is not None:
            output = torch.clip(output, *clip_range)
        # handle alpha
        if res.alpha is not None:
            alpha_tensor = torch.from_numpy(res.alpha)
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
        self.empty_cuda_cache()
        return output

    @classmethod
    def from_esr(
        cls,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> "TranslatorAPI":
        return cls.from_pipeline(esr(), device, use_amp=use_amp, use_half=use_half)

    @classmethod
    def from_esr_anime(
        cls,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> "TranslatorAPI":
        m = esr_anime()
        return cls.from_pipeline(m, device, use_amp=use_amp, use_half=use_half)


def esr(pretrained: bool = True) -> TPipeline:
    return DLZoo.load_pipeline("sr/esr", pretrained=pretrained)


def esr_anime(pretrained: bool = True) -> TPipeline:
    return DLZoo.load_pipeline("sr/esr.anime", pretrained=pretrained)


__all__ = [
    "TranslatorAPI",
]
