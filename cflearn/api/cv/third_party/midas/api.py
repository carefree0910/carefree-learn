# based on https://github.com/isl-org/MiDaS

import math
import torch

import numpy as np
import torch.nn as nn

from torchvision.transforms import Compose

from .core.dpt_depth import DPTDepthModel
from .core.midas_net import MidasNet
from .core.midas_net_custom import MidasNet_small
from .core.transforms import Resize, NormalizeImage, PrepareForNet
from .....misc.toolkit import download_model

try:
    import cv2
except:
    cv2 = None
try:
    import timm
except:
    timm = None


ISL_TAGS = {
    "dpt_large": "dpt_large-midas-2f21e586",
    "dpt_hybrid": "dpt_hybrid-midas-501f0c75",
    "midas_v21": "",
    "midas_v21_small": "",
}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def load_midas_transform(model_type):
    # https://github.com/isl-org/MiDaS/blob/master/run.py
    # load transform only
    if model_type == "dpt_large":  # DPT-Large
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "midas_v21":
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif model_type == "midas_v21_small":
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type large"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    return transform


def load_model(model_type):
    # https://github.com/isl-org/MiDaS/blob/master/run.py
    # load network
    model_tag = ISL_TAGS[model_type]
    model_path = download_model(model_tag)
    if model_type == "dpt_large":  # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "midas_v21":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif model_type == "midas_v21_small":
        model = MidasNet_small(
            model_path,
            features=64,
            backbone="efficientnet_lite3",
            exportable=True,
            non_negative=True,
            blocks={"expand": True},
        )
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    return model.eval(), transform


class MiDaSInference(nn.Module):
    MODEL_TYPES_TORCH_HUB = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
    MODEL_TYPES_ISL = [
        "dpt_large",
        "dpt_hybrid",
        "midas_v21",
        "midas_v21_small",
    ]

    def __init__(self, model_type):
        super().__init__()
        assert model_type in self.MODEL_TYPES_ISL
        model, _ = load_model(model_type)
        self.model = model
        self.model.train = disabled_train

    def forward(self, x):
        with torch.no_grad():
            prediction = self.model(x)
        return prediction


class MiDaSAPI:
    def __init__(self, device: torch.device):
        if cv2 is None:
            raise ValueError("`cv2` is needed for `MiDaSAPI`")
        if timm is None:
            raise ValueError("`timm` is needed for `MiDaSAPI`")
        self.model = MiDaSInference(model_type="dpt_hybrid").to(device)
        self.device = device

    @property
    def dtype(self) -> torch.dtype:
        return list(self.model.parameters())[0].dtype

    def to(self, device: torch.device, *, use_half: bool) -> None:
        if use_half:
            self.model.half()
        else:
            self.model.float()
        self.device = device
        self.model.to(device)

    def to_self(self, net: torch.Tensor) -> torch.Tensor:
        return net.to(self.dtype).to(self.device)

    def detect_depth(self, input_image: np.ndarray) -> np.ndarray:
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = self.to_self(torch.from_numpy(image_depth))
            image_depth = image_depth / 127.5 - 1.0
            image_depth = image_depth[None].permute(0, 3, 1, 2).contiguous()
            depth = self.model(image_depth)[0]

            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_image = depth_pt.cpu().numpy()
            depth_image = (depth_image * 255.0).clip(0, 255).astype(np.uint8)

            return depth_image

    def detect_normal(
        self,
        input_image: np.ndarray,
        a: float = math.pi * 2.0,
        bg_th: float = 0.1,
    ) -> np.ndarray:
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = self.to_self(torch.from_numpy(image_depth))
            image_depth = image_depth / 127.5 - 1.0
            image_depth = image_depth[None].permute(0, 3, 1, 2).contiguous()
            depth = self.model(image_depth)[0]

            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()

            depth_np = depth.cpu().numpy()
            x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
            y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
            z = np.ones_like(x) * a
            x[depth_pt < bg_th] = 0
            y[depth_pt < bg_th] = 0
            normal = np.stack([x, y, z], axis=2)
            normal /= np.sum(normal**2.0, axis=2, keepdims=True) ** 0.5
            normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

            return normal_image


__all__ = [
    "MiDaSAPI",
]
