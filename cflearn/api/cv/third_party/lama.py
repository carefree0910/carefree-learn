import torch

import numpy as np

from abc import abstractmethod
from enum import Enum
from numpy import ndarray
from typing import Any
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from typing import NamedTuple
from PIL.Image import Image
from cftool.cv import to_uint8
from cftool.cv import read_image

from ....misc.toolkit import download_model

try:
    import cv2
except:
    cv2 = None


def ceil_mod(x: int, mod: int) -> int:
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_mod(
    img: ndarray,
    mod: int,
    square: bool = False,
    min_size: Optional[int] = None,
) -> np.array:
    if len(img.shape) == 2:
        img = img[..., None]
    height, width = img.shape[:2]
    out_height = ceil_mod(height, mod)
    out_width = ceil_mod(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def boxes_from_mask(mask: ndarray) -> List[ndarray]:
    height, width = mask.shape[:2]
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box = np.array([x, y, x + w, y + h]).astype(int)
        box[::2] = np.clip(box[::2], 0, width)
        box[1::2] = np.clip(box[1::2], 0, height)
        boxes.append(box)

    return boxes


def resize_max_size(
    np_img: ndarray,
    size_limit: int,
    interpolation: int = 0 if cv2 is None else cv2.INTER_CUBIC,
) -> ndarray:
    # Resize image's longer size to size_limit if longer size larger than size_limit
    h, w = np_img.shape[:2]
    if max(h, w) > size_limit:
        ratio = size_limit / max(h, w)
        new_w = int(w * ratio + 0.5)
        new_h = int(h * ratio + 0.5)
        return cv2.resize(np_img, dsize=(new_w, new_h), interpolation=interpolation)
    else:
        return np_img


def to_tensor(np_img: ndarray) -> torch.Tensor:
    net = np_img.transpose([2, 0, 1])[None]
    net = torch.from_numpy(net)
    return net


class HDStrategy(str, Enum):
    ORIGINAL = "original"
    RESIZE = "resize"
    CROP = "crop"


class Config(NamedTuple):
    hd_strategy: HDStrategy = HDStrategy.CROP
    hd_strategy_resize_limit: int = 2048
    hd_strategy_crop_trigger_size: int = 1280
    hd_strategy_crop_margin: int = 196


class InpaintModel:
    min_size: Optional[int] = None
    pad_mod: int = 8
    pad_to_square: bool = False

    def __init__(self, device: torch.device, **kwargs: Any):
        self.device = device
        self.init_model(device, **kwargs)

    @abstractmethod
    def init_model(self, device: torch.device, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def forward(
        self,
        image: ndarray,
        mask: ndarray,
        config: Config,
    ) -> ndarray:
        pass

    def _pad_and_run(self, image: ndarray, mask: ndarray, config: Config) -> ndarray:
        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_mod(
            image,
            mod=self.pad_mod,
            square=self.pad_to_square,
            min_size=self.min_size,
        )
        pad_mask = pad_img_to_mod(
            mask,
            mod=self.pad_mod,
            square=self.pad_to_square,
            min_size=self.min_size,
        )

        result = self.forward(pad_image, pad_mask, config)
        result = result[:origin_height, :origin_width, :]

        result, image, mask = self.forward_post_process(result, image, mask, config)

        mask = mask[..., None]
        result = result * mask + image * (1.0 - mask)
        return result

    def forward_post_process(
        self,
        result: ndarray,
        image: ndarray,
        mask: ndarray,
        config: Config,
    ) -> Tuple[ndarray, ndarray, ndarray]:
        return result, image, mask

    @torch.no_grad()
    def __call__(self, image: ndarray, mask: ndarray, config: Config) -> ndarray:
        size_limit = config.hd_strategy_crop_trigger_size
        if max(image.shape) <= size_limit:
            result = self._pad_and_run(image, mask, config)
        else:
            if config.hd_strategy == HDStrategy.CROP:
                boxes = boxes_from_mask(to_uint8(mask))
                crop_result = []
                for box in boxes:
                    crop_image, crop_box = self._run_box(image, mask, box, config)
                    crop_result.append((crop_image, crop_box))
                result = image
                for crop_image, crop_box in crop_result:
                    x1, y1, x2, y2 = crop_box
                    result[y1:y2, x1:x2, :] = crop_image
            elif config.hd_strategy == HDStrategy.RESIZE:
                origin_size = image.shape[:2]
                downsize_image = resize_max_size(image, size_limit=size_limit)
                downsize_mask = resize_max_size(mask, size_limit=size_limit)
                result = self._pad_and_run(downsize_image, downsize_mask, config)
                result = cv2.resize(
                    result,
                    (origin_size[1], origin_size[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
                original_pixel_indices = mask < 127
                result[original_pixel_indices] = image[original_pixel_indices]
            else:
                msg = f"unrecognized hd_strategy `{config.hd_strategy}` occurred"
                raise ValueError(msg)
        torch.cuda.empty_cache()
        return result

    def _crop_box(
        self,
        image: ndarray,
        mask: ndarray,
        box: ndarray,
        config: Config,
    ) -> Tuple[ndarray, ndarray, List[int]]:
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        img_h, img_w = image.shape[:2]

        w = box_w + config.hd_strategy_crop_margin * 2
        h = box_h + config.hd_strategy_crop_margin * 2

        _l = cx - w // 2
        _r = cx + w // 2
        _t = cy - h // 2
        _b = cy + h // 2

        l = max(_l, 0)
        r = min(_r, img_w)
        t = max(_t, 0)
        b = min(_b, img_h)

        if _l < 0:
            r += abs(_l)
        if _r > img_w:
            l -= _r - img_w
        if _t < 0:
            b += abs(_t)
        if _b > img_h:
            t -= _b - img_h

        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)

        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]

        return crop_img, crop_mask, [l, t, r, b]

    def _run_box(
        self,
        image: ndarray,
        mask: ndarray,
        box: ndarray,
        config: Config,
    ) -> Tuple[ndarray, List[int]]:
        crop_img, crop_mask, [l, t, r, b] = self._crop_box(image, mask, box, config)
        return self._pad_and_run(crop_img, crop_mask, config), [l, t, r, b]


class LaMa(InpaintModel):
    pad_mod = 8

    def init_model(self, device: torch.device, **kwargs: Any) -> None:
        if cv2 is None:
            raise ValueError("`cv2` is needed for `LaMa`")
        model_path = download_model("lama")
        model = torch.jit.load(model_path, map_location="cpu")
        model = model.to(device)
        model.eval()
        self.model = model
        self.model_path = model_path

    def to(self, device: torch.device) -> None:
        self.device = device
        self.model.to(device)

    def forward(self, image: ndarray, mask: ndarray, config: Config) -> ndarray:
        image, mask = map(to_tensor, [image, mask])
        image = image.to(self.device)
        mask = mask.to(self.device)
        net = self.model(image, mask)
        net = net[0].permute(1, 2, 0).cpu().numpy()
        return net


class LaMaAPI:
    def __init__(self, device: torch.device) -> None:
        self.lama = LaMa(device)

    def inpaint(self, image: Union[str, Image], mask: Union[str, Image]) -> np.ndarray:
        cfg = Config()
        image_arr = read_image(
            image,
            None,
            anchor=None,
            to_torch_fmt=False,
        ).image
        mask_arr = read_image(
            mask,
            None,
            anchor=None,
            to_mask=True,
            to_torch_fmt=False,
        ).image
        mask_arr[mask_arr > 0.0] = 1.0
        return self.lama(image_arr, mask_arr, cfg)


__all__ = [
    "LaMaAPI",
]
