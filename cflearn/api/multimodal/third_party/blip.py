import torch

from PIL import Image
from cftool.cv import to_rgb

from ...common import IAPI
from ....schema import device_type

try:
    from lavis.models import load_model_and_preprocess
except:
    load_model_and_preprocess = None


class BLIPAPI(IAPI):
    def __init__(
        self,
        device: device_type = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        force_not_lazy: bool = False
    ):
        if load_model_and_preprocess is None:
            raise RuntimeError("`salesforce-lavis` is needed for `BLIPAPI`")
        kw = dict(
            name="blip_caption",
            model_type="base_coco",
            is_eval=True,
            device=device,
        )
        m, self.processors, _ = load_model_and_preprocess(**kw)
        super().__init__(
            m,
            device,
            use_amp=use_amp,
            use_half=use_half,
            force_not_lazy=force_not_lazy,
        )

    @torch.no_grad()
    def caption(self, image: Image.Image) -> str:
        rgb_image = to_rgb(image)
        net = self.processors["eval"](rgb_image)[None].to(self.device, self.dtype)
        return self.m.generate(dict(image=net))[0]


__all__ = [
    "BLIPAPI",
]
