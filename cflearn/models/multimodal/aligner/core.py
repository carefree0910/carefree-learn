import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from abc import ABCMeta
from PIL import Image
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from torchvision import transforms
from cftool.misc import check_requires

from .noises import noises
from ..protocol import PerceptorProtocol
from ...bases import ICustomLossModule
from ...bases import ICustomLossOutput
from ....protocol import ModelProtocol
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ...cv.encoder import EncoderBase
from ...nlp.tokenizers import TokenizerProtocol
from ....misc.toolkit import interpolate
from ....misc.toolkit import download_model
from ....misc.toolkit import DropNoGradStatesMixin

try:
    from cfcv.misc.toolkit import to_rgb
except:
    to_rgb = None


class CutOuts(nn.Module):
    def __init__(
        self,
        cut_size: int,
        num_cuts: int,
        aspect_ratio: float,
        noise_factor: float = 0.1,
    ):
        super().__init__()
        self.cut_size = cut_size
        self.num_cuts = num_cuts
        self.aspect_ratio = aspect_ratio
        self.num_cuts_zoom = int(2 * num_cuts / 3)
        self.noise_factor = noise_factor

        self.aug_zoom = transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.4, p=0.7),
                transforms.RandomResizedCrop(
                    size=cut_size,
                    scale=(0.1, 0.75),
                    ratio=(0.85, 1.2),
                ),
                transforms.ColorJitter(0.1, 0.1, 0.1),
            ]
        )

        blocks = []
        if aspect_ratio == 1:
            n_s = 0.95
            n_t = 0.5 * (1.0 - n_s)
            blocks.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=(n_t, n_t),
                    scale=(n_s, n_s),
                )
            )
        elif aspect_ratio > 1:
            n_s = 1.0 / aspect_ratio
            n_t = 0.5 * (1.0 - n_s)
            blocks.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0, n_t),
                    scale=(0.9 * n_s, n_s),
                )
            )
        else:
            n_s = aspect_ratio
            n_t = 0.5 * (1.0 - n_s)
            blocks.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=(n_t, 0),
                    scale=(0.9 * n_s, n_s),
                )
            )
        blocks.extend(
            [
                transforms.RandomPerspective(distortion_scale=0.2, p=0.7),
                transforms.CenterCrop(size=cut_size),
                transforms.ColorJitter(0.1, 0.1, 0.1),
            ]
        )
        self.aug_wide = transforms.Compose(blocks)

        self.avg_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, net: Tensor) -> Tensor:
        cutouts = []
        for _ in range(self.num_cuts):
            cutout = 0.5 * (self.avg_pool(net) + self.max_pool(net))
            if self.aspect_ratio != 1:
                if self.aspect_ratio > 1:
                    cutout = interpolate(cutout, factor=(1.0, self.aspect_ratio))
                else:
                    cutout = interpolate(cutout, factor=(1.0 / self.aspect_ratio, 1.0))
            cutouts.append(cutout)
        net1 = self.aug_zoom(torch.cat(cutouts[: self.num_cuts_zoom], dim=0))
        net2 = self.aug_wide(torch.cat(cutouts[self.num_cuts_zoom :], dim=0))
        net = torch.cat([net1, net2])
        if self.noise_factor > 0.0:
            factors = net.new_empty([self.num_cuts, 1, 1, 1])
            factors.uniform_(0, self.noise_factor)
            net = net + factors * torch.randn_like(net)
        return net


class Aligner(ICustomLossModule, metaclass=ABCMeta):
    perceptor: PerceptorProtocol

    def __init__(
        self,
        perceptor: str,
        generator: str,
        resolution: Tuple[int, int] = (400, 224),
        *,
        num_cuts: int = 36,
        perceptor_config: Optional[Dict[str, Any]] = None,
        generator_config: Optional[Dict[str, Any]] = None,
        perceptor_pretrained_name: Optional[str] = None,
        generator_pretrained_name: Optional[str] = None,
        perceptor_pretrained_path: Optional[str] = None,
        generator_pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.resolution = resolution
        if perceptor_pretrained_path is None:
            if perceptor_pretrained_name is not None:
                perceptor_pretrained_path = download_model(perceptor_pretrained_name)
        if generator_pretrained_path is None:
            if generator_pretrained_name is not None:
                generator_pretrained_path = download_model(generator_pretrained_name)
        self.perceptor = PerceptorProtocol.make(perceptor, perceptor_config or {})
        self.generator = ModelProtocol.make(generator, generator_config or {})
        if not hasattr(self.generator, "encode"):
            raise ValueError(
                "`generator` should implement the `encode` method, because "
                "`Aligner` will apply optimization on the 'encoded' latent map"
            )
        self.perceptor.requires_grad_(False).eval()
        self.generator.requires_grad_(False).eval()
        with torch.no_grad():
            if perceptor_pretrained_path is not None:
                self.perceptor.load_state_dict(torch.load(perceptor_pretrained_path))
            if generator_pretrained_path is not None:
                self.generator.load_state_dict(torch.load(generator_pretrained_path))
        aspect_ratio = resolution[0] / resolution[1]
        self.cutouts = CutOuts(self.perceptor.img_size, num_cuts, aspect_ratio)


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        inp, inp_min, inp_max = args
        ctx.min = inp_min
        ctx.max = inp_max
        ctx.save_for_backward(inp)
        return inp.clamp(inp_min, inp_max)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (inp,) = ctx.saved_tensors
        grad = grad_outputs[0]
        return grad * (grad * (inp - inp.clamp(ctx.min, ctx.max)) >= 0), None, None


class Text2ImageAligner(DropNoGradStatesMixin, Aligner, metaclass=ABCMeta):
    vision: Optional[EncoderBase]

    def __init__(
        self,
        perceptor: str,
        generator: str,
        tokenizer: str,
        resolution: Tuple[int, int] = (400, 224),
        *,
        text: str,
        noise: str = "fractal",
        condition_path: Optional[str] = None,
        vision_encoder: Optional[str] = None,
        vision_encoder_config: Optional[Dict[str, Any]] = None,
        vision_encoder_pretrained_name: Optional[str] = None,
        vision_encoder_pretrained_path: Optional[str] = None,
        vision_weight: float = 0.1,
        vision_monitor_step: int = 5,
        tokenizer_config: Optional[Dict[str, Any]] = None,
        num_cuts: int = 36,
        perceptor_config: Optional[Dict[str, Any]] = None,
        generator_config: Optional[Dict[str, Any]] = None,
        perceptor_pretrained_name: Optional[str] = None,
        generator_pretrained_name: Optional[str] = None,
        perceptor_pretrained_path: Optional[str] = None,
        generator_pretrained_path: Optional[str] = None,
    ):
        if to_rgb is None:
            raise ValueError("`carefree-cv` is needed for `Text2ImageAligner`")
        super().__init__(
            perceptor,
            generator,
            resolution,
            num_cuts=num_cuts,
            perceptor_config=perceptor_config,
            generator_config=generator_config,
            perceptor_pretrained_name=perceptor_pretrained_name,
            generator_pretrained_name=generator_pretrained_name,
            perceptor_pretrained_path=perceptor_pretrained_path,
            generator_pretrained_path=generator_pretrained_path,
        )
        # initialize latent map
        if condition_path is not None:
            img = to_rgb(Image.open(condition_path))
            img = img.resize(resolution, Image.LANCZOS)
        else:
            noise_arr = noises[noise](*resolution)
            img = to_rgb(Image.fromarray(noise_arr))
        img_arr = np.array(img.resize(resolution, Image.LANCZOS))
        img_tensor = torch.from_numpy(img_arr.transpose([2, 0, 1])[None, ...])
        self.initial_img = img_tensor.to(torch.float32) / 255.0
        # we assume that the `generator` requires images between [-1, 1]
        latent = self.generator.encode(self.initial_img * 2.0 - 1.0)
        self.z = nn.Parameter(latent)
        # vision constraints
        self.vision = None
        self.vision_weight = vision_weight
        self.vision_monitor_step = vision_monitor_step
        if vision_encoder is not None:
            if condition_path is None:
                fmt = "`{}` should be provided when `{}` is used"
                raise ValueError(fmt.format("condition_path", "vision_encoder"))
            vepp = vision_encoder_pretrained_path
            if vepp is None:
                if vepp is not None:
                    vepp = download_model(vision_encoder_pretrained_name)
            if vision_encoder_config is None:
                vision_encoder_config = {}
            encoder_base = EncoderBase.get(vision_encoder)
            if check_requires(encoder_base, "img_size"):
                vision_encoder_config["img_size"] = resolution
            vision_encoder_config["in_channels"] = self.initial_img.shape[1]
            self.vision = EncoderBase.make(vision_encoder, vision_encoder_config)
            self.vision.requires_grad_(False).eval()
            if vepp is not None:
                with torch.no_grad():
                    self.vision.load_state_dict(torch.load(vepp))
        # vision latent code
        if self.vision is None:
            self.img_code = None
        else:
            img_code = self.vision.encode({INPUT_KEY: self.initial_img})
            self.register_buffer("img_code", img_code)
        # text latent code
        tokenizer_ins = TokenizerProtocol.make(tokenizer, tokenizer_config or {})
        text_tensor = torch.from_numpy(tokenizer_ins.tokenize(text))
        text_code = self.perceptor.encode_text(text_tensor)
        self.register_buffer("text_code", text_code)

    @abstractmethod
    def generate_raw(self) -> Tensor:
        """should generate an image between [-1, 1]"""

    @abstractmethod
    def perceptor_normalize(self, image: Tensor) -> Tensor:
        pass

    def get_losses(self, trainer: Any) -> ICustomLossOutput:  # type: ignore
        state = trainer.state
        net = self.forward()
        cutouts = self.cutouts(net)
        perceptor_net = self.perceptor_normalize(cutouts)
        img_code = self.perceptor.encode_image(perceptor_net)
        img_code = img_code.unsqueeze(1)
        distances = img_code.sub(self.text_code).norm(dim=2)
        loss = align_loss = (2.0 * ((0.5 * distances).arcsin() ** 2)).mean()
        losses = {"align": align_loss}
        if self.vision is not None and state.step % self.vision_monitor_step == 0:
            img_code = self.vision.encode({INPUT_KEY: net})
            vision_loss = losses["vision"] = F.mse_loss(img_code, self.img_code)
            loss = loss + vision_loss * self.vision_weight
        losses[LOSS_KEY] = loss
        return ICustomLossOutput({PREDICTIONS_KEY: net}, losses)

    # api

    def generate(self) -> Tensor:
        raw = 0.5 * (self.generate_raw() + 1.0)
        clamped = ClampWithGrad.apply(raw, 0.0, 1.0)
        return clamped

    def forward(self) -> Tensor:  # type: ignore
        return self.generate()


__all__ = ["Text2ImageAligner"]
