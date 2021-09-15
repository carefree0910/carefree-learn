from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from torchvision.transforms import Normalize

from .core import Text2ImageAligner
from ..clip import CLIP
from ...cv.generator import VQGANGenerator


@Text2ImageAligner.register("clip_vqgan_aligner")
class CLIPWithVQGANAligner(Text2ImageAligner):
    perceptor: CLIP
    generator: VQGANGenerator

    def __init__(
        self,
        perceptor: str = "clip",
        generator: str = "vqgan_generator",
        tokenizer: str = "clip",
        resolution: Tuple[int, int] = (400, 224),
        *,
        text: str,
        noise: str = "fractal",
        condition_path: Optional[str] = None,
        tokenizer_config: Optional[Dict[str, Any]] = None,
        num_cuts: int = 36,
        perceptor_config: Optional[Dict[str, Any]] = None,
        generator_config: Optional[Dict[str, Any]] = None,
        perceptor_pretrained_name: Optional[str] = "clip",
        generator_pretrained_name: Optional[str] = "vqgan_generator",
        perceptor_pretrained_path: Optional[str] = None,
        generator_pretrained_path: Optional[str] = None,
    ):
        if generator_config is None:
            generator_config = {}
        generator_config.setdefault("img_size", 256)
        super().__init__(
            perceptor,
            generator,
            tokenizer,
            resolution,
            text=text,
            noise=noise,
            condition_path=condition_path,
            tokenizer_config=tokenizer_config,
            num_cuts=num_cuts,
            perceptor_config=perceptor_config,
            generator_config=generator_config,
            perceptor_pretrained_name=perceptor_pretrained_name,
            generator_pretrained_name=generator_pretrained_name,
            perceptor_pretrained_path=perceptor_pretrained_path,
            generator_pretrained_path=generator_pretrained_path,
        )
        self.normalize = Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )

    def generate_raw(self) -> Tensor:
        z_q = self.generator.codebook(self.z)[0]
        return self.generator.decode(z_q, resize=False)

    def normalize_image(self, image: Tensor) -> Tensor:
        return self.normalize(image)


__all__ = [
    "CLIPWithVQGANAligner",
]
