import torch

from PIL import Image

try:
    from lavis.models import load_model_and_preprocess
except:
    load_model_and_preprocess = None
try:
    from cfcv.misc.toolkit import to_rgb
except:
    to_rgb = None


class BLIPAPI:
    def __init__(self, device: torch.device) -> None:
        if to_rgb is None:
            raise ValueError("`carefree-cv` is needed for `BLIPAPI`")
        if load_model_and_preprocess is None:
            raise ValueError("`salesforce-lavis` is needed for `BLIPAPI`")
        kw = dict(
            name="blip_caption",
            model_type="base_coco",
            is_eval=True,
            device=device,
        )
        self.device = device
        self.model, self.processors, _ = load_model_and_preprocess(**kw)

    def to(self, device: torch.device) -> None:
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def caption(self, image: Image.Image) -> str:
        rgb_image = to_rgb(image)
        net = self.processors["eval"](rgb_image)[None].to(self.device)
        return self.model.generate(dict(image=net))[0]


__all__ = [
    "BLIPAPI",
]
