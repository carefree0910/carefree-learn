import os
import json

import numpy as np

from PIL import Image
from tqdm import tqdm
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from cftool.misc import print_warning

from ..basic import CVDataset
from ..basic import Transforms
from ..basic import ImageFolderData
from ..basic import InferenceImageFolderData
from ..basic import ImageFolderDataset
from ..basic import InferenceImageFolderDataset
from ....types import sample_weights_type
from ....constants import INPUT_KEY
from ....constants import STYLE_KEY

try:
    from cfcv.misc.toolkit import to_rgb
except:
    to_rgb = None


class StyleTransferMixin:
    transform: Transforms
    style_paths_file: str = "valid_paths.json"

    def _init_style_paths(self, style_folder: str) -> None:
        self.style_folder = style_folder
        walked = list(os.walk(style_folder))
        extensions = {".jpg", ".png"}
        style_paths_path = os.path.join(style_folder, self.style_paths_file)
        if os.path.isfile(style_paths_path):
            with open(style_paths_path, "r") as f:
                self.style_paths = json.load(f)
            return None
        self.style_paths = []
        for folder, _, files in tqdm(walked, desc="folders", position=0):
            for file in tqdm(files, desc="files", position=1, leave=False):
                if not any(file.endswith(ext) for ext in extensions):
                    continue
                path = os.path.join(folder, file)
                try:
                    to_rgb(Image.open(path)).verify()
                    self.style_paths.append(path)
                except Exception as err:
                    print_warning(f"error occurred ({err}) when reading '{path}'")
                    continue
        with open(style_paths_path, "w") as f:
            json.dump(self.style_paths, f)

    def _inject_style(self, index: int, sample: Dict[str, Any]) -> None:
        style_path = self.style_paths[index % len(self.style_paths)]
        style_img = to_rgb(Image.open(style_path))
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
        extra_label_names: Optional[List[str]] = None,
        lmdb_config: Optional[Dict[str, Any]] = None,
        *,
        style_folder: str,
    ):
        if to_rgb is None:
            raise ValueError("`carefree-cv` is needed for `StyleTransferDataset`")
        super().__init__(
            folder,
            split,
            transform,
            transform_config,
            extra_label_names,
            lmdb_config,
        )
        self._init_style_paths(style_folder)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)
        self._inject_style(index, sample)
        return sample


class InferenceStyleTransferDataset(InferenceImageFolderDataset, StyleTransferMixin):  # type: ignore
    def __init__(
        self,
        folder: str,
        transform: Optional[Transforms],
        *,
        style_folder: str,
    ):
        if to_rgb is None:
            msg = "`carefree-cv` is needed for `InferenceStyleTransferDataset`"
            raise ValueError(msg)
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
        drop_train_last: bool = True,
        transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
        transform_config: Optional[Dict[str, Any]] = None,
        test_shuffle: Optional[bool] = None,
        test_transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
        test_transform_config: Optional[Dict[str, Any]] = None,
        lmdb_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            folder,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_train_last=drop_train_last,
            transform=transform,
            transform_config=transform_config,
            test_shuffle=test_shuffle,
            test_transform=test_transform,
            test_transform_config=test_transform_config,
            lmdb_config=lmdb_config,
        )
        self.style_folder = style_folder

    @property
    def info(self) -> Dict[str, Any]:
        d = super().info
        d["style_folder"] = self.style_folder
        return d

    def prepare(self, sample_weights: sample_weights_type) -> None:
        self.train_data = CVDataset(
            StyleTransferDataset(
                self.folder,
                "train",
                self.transform,
                lmdb_config=self.lmdb_config,
                style_folder=self.style_folder,
            )
        )
        self.setup_train_weights()
        self.valid_data = CVDataset(
            StyleTransferDataset(
                self.folder,
                "valid",
                self.test_transform,
                lmdb_config=self.lmdb_config,
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
