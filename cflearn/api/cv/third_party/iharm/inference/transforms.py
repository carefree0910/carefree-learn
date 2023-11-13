import torch

from numpy import ndarray
from torch import Tensor
from typing import Tuple
from collections import namedtuple

try:
    import cv2
except:
    cv2 = None


APack = Tuple[ndarray, ndarray]
TPack = Tuple[Tensor, Tensor]


class EvalTransform:
    def __init__(self):
        pass

    def transform(self, image: Tensor, mask: Tensor) -> TPack:
        raise NotImplementedError

    def inv_transform(self, image: Tensor) -> Tensor:
        raise NotImplementedError


class PadToDivisor(EvalTransform):
    """
    Pad side of the image so that its side is divisible by divisor.

    Args:
        divisor (int): desirable image size divisor
        border_mode (OpenCV flag): OpenCV border mode.
        fill_value (int, float, list of int, lisft of float): padding value if border_mode is cv2.BORDER_CONSTANT.
    """

    PadParams = namedtuple("PadParams", ["top", "bottom", "left", "right"])

    def __init__(
        self,
        divisor: int,
        border_mode: int = 0 if cv2 is None else cv2.BORDER_CONSTANT,
        fill_value: int = 0,
    ):
        super().__init__()
        self.border_mode = border_mode
        self.fill_value = fill_value
        self.divisor = divisor
        self._pads = None

    def transform(self, image: ndarray, mask: ndarray) -> APack:
        if cv2 is None:
            raise ValueError("`cv2` is needed for `PadToDivisor`")

        self._pads = PadToDivisor.PadParams(
            *self._get_dim_padding(image.shape[0]),
            *self._get_dim_padding(image.shape[1]),
        )

        image = cv2.copyMakeBorder(
            image,
            *self._pads,
            self.border_mode,
            value=self.fill_value,
        )
        mask = cv2.copyMakeBorder(
            mask,
            *self._pads,
            self.border_mode,
            value=self.fill_value,
        )
        return image, mask

    def inv_transform(self, image: Tensor) -> Tensor:
        assert (
            self._pads is not None
        ), "Something went wrong, inv_transform(...) should be called after transform(...)"
        return self._remove_padding(image)

    def _get_dim_padding(self, dim_size: int) -> Tuple[int, int]:
        pad = (self.divisor - dim_size % self.divisor) % self.divisor
        pad_upper = pad // 2
        pad_lower = pad - pad_upper
        return pad_upper, pad_lower

    def _remove_padding(self, tensor: Tensor) -> Tensor:
        tensor_h, tensor_w = tensor.shape[:2]
        cropped = tensor[
            self._pads.top : tensor_h - self._pads.bottom,
            self._pads.left : tensor_w - self._pads.right,
            :,
        ]
        return cropped


class NormalizeTensor(EvalTransform):
    def __init__(self, mean: float, std: float, device: torch.device):
        super().__init__()
        self.mean = torch.as_tensor(mean).reshape(1, 3, 1, 1).to(device)
        self.std = torch.as_tensor(std).reshape(1, 3, 1, 1).to(device)

    def transform(self, image: Tensor, mask: Tensor) -> TPack:
        image.sub_(self.mean).div_(self.std)
        return image, mask

    def inv_transform(self, image: Tensor) -> Tensor:
        image.mul_(self.std).add_(self.mean)
        return image


class ToTensor(EvalTransform):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def transform(self, image: ndarray, mask: ndarray) -> TPack:
        image = torch.as_tensor(image, device=self.device, dtype=torch.float32)
        mask = torch.as_tensor(mask, device=self.device)
        image.unsqueeze_(0)
        mask.unsqueeze_(0).unsqueeze_(0)
        return image.permute(0, 3, 1, 2) / 255.0, mask

    def inv_transform(self, image: Tensor) -> Tensor:
        image.squeeze_(0)
        return 255 * image.permute(1, 2, 0)


class AddFlippedTensor(EvalTransform):
    def transform(self, image: Tensor, mask: Tensor) -> TPack:
        flipped_image = torch.flip(image, dims=(3,))
        flipped_mask = torch.flip(mask, dims=(3,))
        image = torch.cat((image, flipped_image), dim=0)
        mask = torch.cat((mask, flipped_mask), dim=0)
        return image, mask

    def inv_transform(self, image: Tensor) -> Tensor:
        return 0.5 * (image[:1] + torch.flip(image[1:], dims=(3,)))
