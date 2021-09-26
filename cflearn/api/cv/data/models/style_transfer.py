import os
import json
import random

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

from .....types import sample_weights_type
from .....constants import INPUT_KEY
from .....constants import LABEL_KEY
from .....constants import WARNING_PREFIX
from .....misc.internal_ import CVDataset
from .....misc.internal_ import Transforms
from .....misc.internal_ import ImageFolderData
from .....misc.internal_ import InferenceImageFolderData
from .....misc.internal_.data.core import ImageFolderDataset
from .....misc.internal_.data.core import InferenceImageFolderDataset
from .....models.cv.stylizer.constants import STYLE_KEY
from .....models.cv.stylizer.constants import INPUT_B_KEY
from .....models.cv.stylizer.constants import LABEL_B_KEY


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
                    Image.open(path).convert("RGB").verify()
                    self.style_paths.append(path)
                except Exception as err:
                    msg = f"error occurred ({err}) when reading '{path}'"
                    print(f"{WARNING_PREFIX}{msg}")
                    continue
        with open(style_paths_path, "w") as f:
            json.dump(self.style_paths, f)

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
        extra_label_names: Optional[List[str]] = None,
        lmdb_config: Optional[Dict[str, Any]] = None,
        *,
        style_folder: str,
    ):
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
        lmdb_config: Optional[Dict[str, Any]] = None,
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
            lmdb_config=lmdb_config,
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
                lmdb_config=self.lmdb_config,
                style_folder=self.style_folder,
            )
        )
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


class UnifiedStyleTransferDataset(ImageFolderDataset):
    def __init__(
        self,
        folder: str,
        split: str,
        transform: Optional[Union[str, List[str], Transforms, Callable]],
        transform_config: Optional[Dict[str, Any]] = None,
        extra_label_names: Optional[List[str]] = None,
        lmdb_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            folder,
            split,
            transform,
            transform_config,
            extra_label_names,
            lmdb_config,
        )
        self.style2paths: Dict[str, List[str]] = {}
        self.style_labels = {}
        self.content_labels = {}
        self.style2content2path: Dict[str, Dict[str, int]] = {}
        with open(os.path.join(self.folder, "paths.json"), "r") as f:
            paths = json.load(f)
        assert set(paths) == set(self.img_paths)
        with open(os.path.join(self.folder, "path_mapping.json"), "r") as f:
            path_mapping = json.load(f)
        for i, path in enumerate(self.img_paths):
            original_path = path_mapping[path]
            original_path = os.path.normpath(original_path)
            split = original_path.split(os.sep)
            content_label, style_label = split[-1], split[-2]
            content_label = os.path.splitext(content_label)[0]
            self.style_labels[path] = style_label
            self.content_labels[path] = content_label
            self.style2paths.setdefault(style_label, []).append(path)
            self.style2content2path.setdefault(style_label, {})[content_label] = i

    def __getitem__(self, index: int) -> Dict[str, Any]:
        net_a = super().__getitem__(index)[INPUT_KEY]
        path_a = self.img_paths[index]
        style_a = self.style_labels[path_a]
        content_a = self.content_labels[path_a]
        content_b = None
        while True:
            b_index = random.randint(0, self.length - 1)
            net_b = super().__getitem__(b_index)[INPUT_KEY]
            path_b = self.img_paths[b_index]
            style_b = self.style_labels[path_b]
            if style_a == style_b:
                continue
            content_b = self.content_labels[path_b]
            if content_a not in self.style2content2path[style_b]:
                continue
            if content_b not in self.style2content2path[style_a]:
                continue
            break
        target_ab_index = self.style2content2path[style_b][content_a]
        target_ba_index = self.style2content2path[style_a][content_b]
        label_ab = super().__getitem__(target_ab_index)[INPUT_KEY]
        label_ba = super().__getitem__(target_ba_index)[INPUT_KEY]
        return {
            INPUT_KEY: net_a,
            INPUT_B_KEY: net_b,
            LABEL_KEY: label_ab,
            LABEL_B_KEY: label_ba,
        }


@ImageFolderData.register("unified_style_transfer")
class UnifiedStyleTransferData(ImageFolderData):
    # TODO : support sample weights
    def prepare(self, sample_weights: sample_weights_type) -> None:
        self.train_data = CVDataset(
            UnifiedStyleTransferDataset(
                self.folder,
                "train",
                self.transform,
                lmdb_config=self.lmdb_config,
            )
        )
        self.valid_data = CVDataset(
            UnifiedStyleTransferDataset(
                self.folder,
                "valid",
                self.test_transform,
                lmdb_config=self.lmdb_config,
            )
        )


class SiameseStyleTransferDataset(ImageFolderDataset):
    def __init__(
        self,
        folder: str,
        split: str,
        transform: Optional[Union[str, List[str], Transforms, Callable]],
        transform_config: Optional[Dict[str, Any]] = None,
        extra_label_names: Optional[List[str]] = None,
        lmdb_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            folder,
            split,
            transform,
            transform_config,
            extra_label_names,
            lmdb_config,
        )
        self.style2paths: Dict[str, List[str]] = {}
        self.style_labels = {}
        with open(os.path.join(self.folder, "paths.json"), "r") as f:
            paths = json.load(f)
        assert set(paths) == set(self.img_paths)
        with open(os.path.join(self.folder, "path_mapping.json"), "r") as f:
            path_mapping = json.load(f)
        for i, path in enumerate(self.img_paths):
            original_path = path_mapping[path]
            original_path = os.path.normpath(original_path)
            style_label = original_path.split(os.sep)[-2]
            self.style_labels[path] = style_label
            self.style2paths.setdefault(style_label, []).append(path)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        inp_a = super().__getitem__(index)
        path_a = self.img_paths[index]
        style_a = self.style_labels[path_a]
        style_paths = self.style2paths[style_a]
        path_b = path_a
        for _ in range(10):
            path_b = random.choice(style_paths)
            if path_b != path_a:
                break
        index_b = self.img_paths.index(path_b)
        inp_b = super().__getitem__(index_b)
        return {
            INPUT_KEY: inp_a[INPUT_KEY],
            INPUT_B_KEY: inp_b[INPUT_KEY],
            LABEL_KEY: inp_a[LABEL_KEY],
            LABEL_B_KEY: inp_b[LABEL_KEY],
        }


@ImageFolderData.register("siamese_style_transfer")
class SiameseStyleTransferData(ImageFolderData):
    # TODO : support sample weights
    def prepare(self, sample_weights: sample_weights_type) -> None:
        self.train_data = CVDataset(
            SiameseStyleTransferDataset(
                self.folder,
                "train",
                self.transform,
                lmdb_config=self.lmdb_config,
            )
        )
        self.valid_data = CVDataset(
            SiameseStyleTransferDataset(
                self.folder,
                "valid",
                self.test_transform,
                lmdb_config=self.lmdb_config,
            )
        )
