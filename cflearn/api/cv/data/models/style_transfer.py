import os

import numpy as np

from PIL import Image
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional

from ..core import Transforms
from ..core import ImageFolderDataset
from ..core import InferenceImageFolderDataset
from ..interface import ImageFolderData
from ..interface import InferenceImageFolderData
from .....types import sample_weights_type
from .....constants import INPUT_KEY
from .....misc.internal_ import CVDataset
from .....models.cv.generator.constants import STYLE_KEY


class StyleTransferMixin:
    transform: Transforms

    def _init_style_paths(self, style_folder: str) -> None:
        self.style_folder = style_folder
        style_files = sorted(os.listdir(style_folder))
        style_paths = [os.path.join(style_folder, file) for file in style_files]
        self.style_paths = list(map(os.path.abspath, style_paths))  # type: ignore

    def _inject_style(self, index: int, sample: Dict[str, Any]) -> None:
        style_path = self.style_paths[index % len(self.style_paths)]
        style_img = Image.open(style_path).convert("RGB")
        style_arr = np.array(style_img).astype(np.float32) / 255.0
        style_tensor = self.transform({INPUT_KEY: style_arr})[INPUT_KEY]
        sample[STYLE_KEY] = style_tensor


class StyleTransferDataset(ImageFolderDataset, StyleTransferMixin):  # type: ignore
    def __init__(
        self,
        folder: str,
        split: str,
        transform: Optional[Union[str, List[str], Transforms, Callable]],
        transform_config: Optional[Dict[str, Any]] = None,
        lmdb_configs: Optional[Dict[str, Any]] = None,
        *,
        style_folder: str,
    ):
        super().__init__(folder, split, transform, transform_config, lmdb_configs)
        self._init_style_paths(style_folder)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)
        self._inject_style(index, sample)
        return sample


class InferenceStyleTransferDataset(InferenceImageFolderDataset, StyleTransferMixin):  # type: ignore
    def __init__(
        self,
        folder: str,
        transform: Optional[Callable],
        *,
        style_folder: str,
    ):
        super().__init__(folder, transform)
        self._init_style_paths(style_folder)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        sample = super().__getitem__(index)
        self._inject_style(index, sample)
        return sample


@ImageFolderData.register("style_transfer")
class StyleTransferData(ImageFolderData):
    def __init__(
        self,
        folder: str,
        style_folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
        transform_config: Optional[Dict[str, Any]] = None,
        test_shuffle: Optional[bool] = None,
        test_transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
        test_transform_config: Optional[Dict[str, Any]] = None,
        lmdb_configs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            folder,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            transform=transform,
            transform_config=transform_config,
            test_shuffle=test_shuffle,
            test_transform=test_transform,
            test_transform_config=test_transform_config,
            lmdb_configs=lmdb_configs,
        )
        self.style_folder = style_folder

    @property
    def info(self) -> Dict[str, Any]:
        d = super().info
        d["style_folder"] = self.style_folder
        return d

    # TODO : support sample weights
    def prepare(self, sample_weights: sample_weights_type) -> None:
        self.train_data = CVDataset(
            StyleTransferDataset(
                self.folder,
                "train",
                self.transform,
                lmdb_configs=self.lmdb_configs,
                style_folder=self.style_folder,
            )
        )
        self.valid_data = CVDataset(
            StyleTransferDataset(
                self.folder,
                "valid",
                self.test_transform,
                lmdb_configs=self.lmdb_configs,
                style_folder=self.style_folder,
            )
        )


class InferenceStyleTransferData(InferenceImageFolderData):
    def __init__(
        self,
        folder: str,
        style_folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
        transform_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            folder,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=transform,
            transform_config=transform_config,
        )
        self.style_folder = self.kw["style_folder"] = style_folder

    def prepare(self, sample_weights: sample_weights_type) -> None:
        self.dataset = InferenceStyleTransferDataset(
            self.folder,
            self.transform,
            style_folder=self.style_folder,
        )
