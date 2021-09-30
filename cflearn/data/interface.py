import os
import dill
import json
import lmdb
import torch
import random
import shutil
import tempfile

import numpy as np

from abc import ABCMeta
from PIL import Image
from tqdm import tqdm
from torch import Tensor
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.dist import Parallel
from cftool.misc import shallow_copy_dict
from cftool.misc import Saving
from torch.utils.data import Dataset
from cfdata.tabular.api import TabularData

from .core import _default_lmdb_path
from .core import CVLoader
from .core import DLLoader
from .core import LMDBItem
from .core import MLLoader
from .core import CVDataset
from .core import DLDataset
from .core import MLDataset
from .core import DataLoader
from .core import Transforms
from .core import DLDataModule
from .core import ImageFolderDataset
from .core import InferenceImageFolderDataset
from ..types import data_type
from ..types import tensor_dict_type
from ..types import sample_weights_type
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..constants import INFO_PREFIX
from ..constants import ERROR_PREFIX
from ..constants import WARNING_PREFIX
from ..constants import ORIGINAL_LABEL_KEY
from ..misc.toolkit import walk


# dl


class TensorDataset(Dataset):
    def __init__(
        self,
        x: Tensor,
        y: Optional[Tensor],
        others: Optional[tensor_dict_type] = None,
    ):
        self.x = x
        self.y = y
        self.others = others

    def __getitem__(self, index: int) -> tensor_dict_type:
        label = 0 if self.y is None else self.y[index]
        item = {
            INPUT_KEY: self.x[index],
            LABEL_KEY: label,
            ORIGINAL_LABEL_KEY: label,
        }
        if self.others is not None:
            for k, v in self.others.items():
                item[k] = v[index]
        return item

    def __len__(self) -> int:
        return self.x.shape[0]


@DLDataModule.register("tensor")
class TensorData(DLDataModule):
    def __init__(
        self,
        x_train: Tensor,
        y_train: Optional[Tensor] = None,
        x_valid: Optional[Tensor] = None,
        y_valid: Optional[Tensor] = None,
        train_others: Optional[tensor_dict_type] = None,
        valid_others: Optional[tensor_dict_type] = None,
        *,
        shuffle: bool = True,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.train_others = train_others
        self.valid_others = valid_others
        self.kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.test_transform = None

    @property
    def info(self) -> Dict[str, Any]:
        return self.kw

    # TODO : support sample weights
    def prepare(self, sample_weights: sample_weights_type) -> None:
        def _get_data(x: Any, y: Any, others: Any) -> DLDataset:
            return DLDataset(TensorDataset(x, y, others))

        self.train_data = _get_data(self.x_train, self.y_train, self.train_others)
        if self.x_valid is None:
            self.valid_data = None
        else:
            self.valid_data = _get_data(self.x_valid, self.y_valid, self.valid_others)

    def initialize(self) -> Tuple[DLLoader, Optional[DLLoader]]:
        train_loader = DLLoader(DataLoader(self.train_data, **self.kw))  # type: ignore
        if self.valid_data is None:
            valid_loader = None
        else:
            valid_loader = DLLoader(DataLoader(self.valid_data, **self.kw))  # type: ignore
        return train_loader, valid_loader


@DLDataModule.register("dummy")
class DummyData(TensorData):
    def __init__(
        self,
        *,
        num_samples: int = 1,
        batch_size: int = 1,
        use_valid: bool = False,
    ):
        dummy = torch.zeros([num_samples, 1])
        x_valid = y_valid = dummy if use_valid else None
        super().__init__(dummy, dummy, x_valid, y_valid, batch_size=batch_size)


# ml


split_sw_type = Tuple[Optional[np.ndarray], Optional[np.ndarray]]


def _norm_sw(sample_weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if sample_weights is None:
        return None
    return sample_weights / sample_weights.sum()


def _split_sw(sample_weights: sample_weights_type) -> split_sw_type:
    if sample_weights is None:
        train_weights = valid_weights = None
    else:
        if not isinstance(sample_weights, np.ndarray):
            train_weights, valid_weights = sample_weights
        else:
            train_weights, valid_weights = sample_weights, None
    train_weights, valid_weights = map(_norm_sw, [train_weights, valid_weights])
    return train_weights, valid_weights


@DLDataModule.register("ml")
class MLData(DLDataModule):
    train_data: MLDataset
    valid_data: Optional[MLDataset]

    tmp_cf_data_name = ".tmp_cf_data"

    def __init__(
        self,
        x_train: data_type,
        y_train: data_type = None,
        x_valid: data_type = None,
        y_valid: data_type = None,
        *,
        cf_data: Optional[TabularData] = None,
        num_history: int = 1,
        is_classification: Optional[bool] = None,
        read_config: Optional[Dict[str, Any]] = None,
        # valid split
        valid_split: Optional[Union[int, float]] = None,
        min_valid_split: int = 100,
        max_valid_split: int = 10000,
        max_valid_split_ratio: float = 0.5,
        valid_split_order: str = "auto",
        # data loader
        shuffle_train: bool = True,
        shuffle_valid: bool = False,
        batch_size: int = 128,
        valid_batch_size: int = 512,
        # inference
        for_inference: bool = False,
    ):
        assert x_train is not None
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.cf_data = cf_data
        self.num_history = num_history
        if is_classification is None and cf_data is None and not for_inference:
            msg = "`cf_data` should be provided when `is_classification` is None"
            raise ValueError(msg)
        self.is_classification = is_classification
        self.read_config = read_config or {}
        self.valid_split = valid_split
        self.min_valid_split = min_valid_split
        self.max_valid_split = max_valid_split
        self.max_valid_split_ratio = max_valid_split_ratio
        self.valid_split_order = valid_split_order
        self.shuffle_train = shuffle_train
        self.shuffle_valid = shuffle_valid
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "cf_data": self.cf_data,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "num_history": self.num_history,
            "is_classification": self.is_classification,
        }

    def prepare(self, sample_weights: sample_weights_type) -> None:
        self.train_weights, self.valid_weights = _split_sw(sample_weights)
        if self.cf_data is not None:
            self.cf_data.read(self.x_train, self.y_train, **self.read_config)
            if self.is_classification is None:
                self.is_classification = self.cf_data.is_clf
            if self.x_valid is not None:
                self.train_cf_data = self.cf_data
                self.valid_cf_data = self.cf_data.copy_to(self.x_valid, self.y_valid)
            else:
                if isinstance(self.valid_split, int):
                    split = self.valid_split
                else:
                    num_data = len(self.cf_data)
                    if isinstance(self.valid_split, float):
                        split = int(round(self.valid_split * num_data))
                    else:
                        default_split = 0.1
                        num_split = int(round(default_split * num_data))
                        num_split = max(self.min_valid_split, num_split)
                        max_split = int(round(num_data * self.max_valid_split_ratio))
                        max_split = min(max_split, self.max_valid_split)
                        split = min(num_split, max_split)
                if split <= 0:
                    self.train_cf_data = self.cf_data
                    self.valid_cf_data = None
                else:
                    rs = self.cf_data.split(split, order=self.valid_split_order)
                    self.train_cf_data = rs.remained
                    self.valid_cf_data = rs.split
            self.train_data = MLDataset(*self.train_cf_data.processed.xy)
            if self.valid_cf_data is None:
                self.valid_data = None
            else:
                self.valid_data = MLDataset(*self.valid_cf_data.processed.xy)
            self.num_classes = self.train_cf_data.num_classes
            self.input_dim = self.train_cf_data.processed_dim
            return None
        if isinstance(self.x_train, str):
            raise ValueError("`cf_data` should be provided when `x_train` is `str`")
        self.num_classes = None
        self.input_dim = self.x_train.shape[-1]
        self.train_data = MLDataset(self.x_train, self.y_train)
        if self.x_valid is None or self.y_valid is None:
            self.valid_data = None
        else:
            self.valid_data = MLDataset(self.x_valid, self.y_valid)

    def initialize(self) -> Tuple[MLLoader, Optional[MLLoader]]:
        train_loader = MLLoader(
            self.train_data,
            name="train",
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
            sample_weights=self.train_weights,
        )
        if self.valid_data is None:
            valid_loader = None
        else:
            valid_loader = MLLoader(
                self.valid_data,
                name="valid",
                shuffle=self.shuffle_valid,
                batch_size=self.valid_batch_size,
                sample_weights=self.valid_weights,
            )
        return train_loader, valid_loader

    def _save_info(self, folder: str) -> None:
        info = self.info
        if info["cf_data"] is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_name = os.path.join(tmp_dir, self.tmp_cf_data_name)
                info["cf_data"].save(tmp_name, retain_data=False)
                zip_file = f"{tmp_name}.zip"
                with open(zip_file, "rb") as f:
                    info["cf_data"] = f.read()
                os.remove(zip_file)
        Saving.save_dict(info, self.info_name, folder)

    @classmethod
    def _load_info(cls, folder: str) -> Dict[str, Any]:
        d = super()._load_info(folder)
        cf_data = d["cf_data"]
        if cf_data is None:
            return d
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_name = os.path.join(tmp_dir, cls.tmp_cf_data_name)
            zip_file = f"{tmp_name}.zip"
            with open(zip_file, "wb") as f:
                f.write(cf_data)
            d["cf_data"] = TabularData.load(tmp_name)
            os.remove(zip_file)
        return d

    @classmethod
    def with_cf_data(
        cls,
        *args: Any,
        is_classification: Optional[bool] = None,
        cf_data_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "MLData":
        if cf_data_config is None:
            cf_data_config = {}
        cf_data_config["default_categorical_process"] = "identical"
        if is_classification is not None:
            cf_data_config["task_type"] = "clf" if is_classification else "reg"
        kwargs["is_classification"] = is_classification
        kwargs["cf_data"] = TabularData(**(cf_data_config or {}))
        return cls(*args, **kwargs)


class MLInferenceData(MLData):
    def __init__(self, x: data_type, y: data_type = None):
        super().__init__(x, y, for_inference=True)


# cv


class CVDataModule(DLDataModule, metaclass=ABCMeta):
    test_transform: Optional[Transforms]


@DLDataModule.register("image_folder")
class ImageFolderData(CVDataModule):
    def __init__(
        self,
        folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        prefetch_device: Optional[Union[int, str]] = None,
        pin_memory_device: Optional[Union[int, str]] = None,
        extra_label_names: Optional[List[str]] = None,
        transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
        transform_config: Optional[Dict[str, Any]] = None,
        test_shuffle: Optional[bool] = None,
        test_transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
        test_transform_config: Optional[Dict[str, Any]] = None,
        lmdb_config: Optional[Dict[str, Any]] = None,
    ):
        self.folder = folder
        self.shuffle = shuffle
        if not torch.cuda.is_available():
            fmt = "cuda is not available but {} is provided, which will have no effect"
            if prefetch_device is not None:
                print(f"{WARNING_PREFIX}{fmt.format('`prefetch_device`')}")
                prefetch_device = None
            if pin_memory_device is not None:
                print(f"{WARNING_PREFIX}{fmt.format('`pin_memory_device`')}")
                pin_memory_device = None
        if prefetch_device is not None:
            if pin_memory_device is None:
                pin_memory_device = prefetch_device
            if pin_memory_device is not None:
                if str(prefetch_device) != str(pin_memory_device):
                    print(
                        f"{WARNING_PREFIX}`prefetch_device` and `pin_memory_device` "
                        "are both provided but they are not consistent, which may "
                        "impact the memory usage and performance"
                    )
        self.prefetch_device = prefetch_device
        self.pin_memory_device = pin_memory_device
        self.extra_label_names = extra_label_names
        self.transform = Transforms.convert(transform, transform_config)
        self.test_shuffle = test_shuffle
        if test_transform is None:
            test_transform = transform
        if test_transform_config is None:
            test_transform_config = transform_config
        self.test_transform = Transforms.convert(test_transform, test_transform_config)
        if self.test_transform is None:
            self.test_transform = self.transform
        self.lmdb_config = lmdb_config
        self.kw = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory_device is not None,
        )

    @property
    def info(self) -> Dict[str, Any]:
        d = shallow_copy_dict(self.kw)
        d["test_shuffle"] = self.test_shuffle
        try:
            json.dumps(self.lmdb_config)
            d["lmdb_config"] = self.lmdb_config
        except Exception as err:
            d["lmdb_config"] = str(err)
        return d

    # TODO : support sample weights
    def prepare(self, sample_weights: sample_weights_type) -> None:
        self.train_data = CVDataset(
            ImageFolderDataset(
                self.folder,
                "train",
                self.transform,
                extra_label_names=self.extra_label_names,
                lmdb_config=self.lmdb_config,
            )
        )
        self.valid_data = CVDataset(
            ImageFolderDataset(
                self.folder,
                "valid",
                self.test_transform,
                extra_label_names=self.extra_label_names,
                lmdb_config=self.lmdb_config,
            )
        )

    def initialize(self) -> Tuple[CVLoader, Optional[CVLoader]]:
        if self.pin_memory_device is not None:
            torch.cuda.set_device(self.pin_memory_device)
        d = shallow_copy_dict(self.kw)
        kw = {"prefetch_device": self.prefetch_device}
        train_loader = CVLoader(DataLoader(self.train_data, **d), **kw)  # type: ignore
        d["shuffle"] = self.test_shuffle or self.shuffle
        valid_loader = CVLoader(DataLoader(self.valid_data, **d), **kw)  # type: ignore
        return train_loader, valid_loader


class InferenceImageFolderData(CVDataModule):
    def __init__(
        self,
        folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        prefetch_device: Optional[Union[int, str]] = None,
        transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
        transform_config: Optional[Dict[str, Any]] = None,
    ):
        self.folder = folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_device = prefetch_device
        self.transform = Transforms.convert(transform, transform_config)
        self.kw: Dict[str, Any] = dict(batch_size=batch_size, num_workers=num_workers)
        self.prepare(None)

    @property
    def info(self) -> Dict[str, Any]:
        return self.kw

    def prepare(self, sample_weights: sample_weights_type) -> None:
        self.dataset = InferenceImageFolderDataset(self.folder, self.transform)

    def initialize(self) -> Tuple[CVLoader, Optional[CVLoader]]:
        args = self.batch_size, self.num_workers, self.prefetch_device
        loader = self.dataset.make_loader(*args)
        return loader, None

    @classmethod
    def from_package(
        cls,
        src_folder: str,
        *,
        package_folder: str,
        batch_size: int,
        num_workers: int = 0,
    ) -> "InferenceImageFolderData":
        info = cls.load(package_folder)
        transform = info["test_transform"]
        return cls(
            src_folder,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=transform,
        )


class PrepareResults(NamedTuple):
    data: ImageFolderData
    tgt_folder: str


class _PreparationProtocol:
    @property
    def extra_labels(self) -> Optional[List[str]]:
        pass

    def prepare_src_folder(self, src_path: str) -> None:
        pass

    def filter(self, hierarchy: List[str]) -> bool:
        pass

    def get_label(self, hierarchy: List[str]) -> Any:
        pass

    def get_extra_label(self, label_name: str, hierarchy: List[str]) -> Any:
        pass

    def copy(self, src_path: str, tgt_path: str) -> None:
        pass

    def get_new_img_path(self, idx: int, split_folder: str, old_img_path: str) -> str:
        pass

    def is_ready(self, tgt_folder: str) -> bool:
        for split in ["train", "valid"]:
            extra_keys = [f"{key}_labels" for key in self.extra_labels or []]
            for key in [LABEL_KEY] + extra_keys:
                path = os.path.join(tgt_folder, split, f"{key}.json")
                if not os.path.isfile(path):
                    return False
        return True

    def get_num_classes(self, tgt_folder: str) -> Dict[str, int]:
        num_classes = {}
        for label_name in [LABEL_KEY] + (self.extra_labels or []):
            with open(os.path.join(tgt_folder, f"idx2{label_name}.json"), "r") as f:
                num_classes[label_name] = len(json.load(f))
        return num_classes


class DefaultPreparation(_PreparationProtocol):
    @property
    def extra_labels(self) -> Optional[List[str]]:
        return None

    def filter(self, hierarchy: List[str]) -> bool:
        return True

    def get_label(self, hierarchy: List[str]) -> Any:
        return 0

    def copy(self, src_path: str, tgt_path: str) -> None:
        shutil.copyfile(src_path, tgt_path)

    def get_new_img_path(self, idx: int, split_folder: str, old_img_path: str) -> str:
        ext = os.path.splitext(old_img_path)[1]
        return os.path.join(split_folder, f"{idx}{ext}")


def prepare_image_folder(
    src_folder: str,
    tgt_folder: str,
    *,
    to_index: bool,
    prefix: Optional[str] = None,
    preparation: _PreparationProtocol = DefaultPreparation(),
    force_rerun: bool = False,
    extensions: Optional[Set[str]] = None,
    make_labels_in_parallel: bool = False,
    num_jobs: int = 8,
    train_all_data: bool = False,
    valid_split: Union[int, float] = 0.1,
    max_num_valid: int = 10000,
    lmdb_config: Optional[Dict[str, Any]] = None,
    use_tqdm: bool = True,
) -> str:
    if prefix is not None:
        src_folder = os.path.join(prefix, src_folder)
        tgt_folder = os.path.join(prefix, tgt_folder)

    if not force_rerun and preparation.is_ready(tgt_folder):
        return tgt_folder

    preparation.prepare_src_folder(src_folder)
    if os.path.isdir(tgt_folder):
        print(f"{WARNING_PREFIX}'{tgt_folder}' already exists, it will be removed")
        shutil.rmtree(tgt_folder)
    os.makedirs(tgt_folder, exist_ok=True)

    print(f"{INFO_PREFIX}collecting hierarchies")

    def hierarchy_callback(hierarchy: List[str], path: str) -> None:
        hierarchy = hierarchy[prefix_idx:]
        if not preparation.filter(hierarchy):
            return None
        hierarchy_list.append(hierarchy)
        all_img_paths.append(path)

    all_img_paths: List[str] = []
    hierarchy_list: List[List[str]] = []
    if extensions is None:
        extensions = {".jpg", ".png"}
    prefix_idx = 0
    if prefix is not None:
        prefix_idx = len(prefix.split(os.path.sep))
    walk(src_folder, hierarchy_callback, extensions)

    def get_labels(
        label_fn: Callable,
        label_name: Optional[str] = None,
    ) -> List[Any]:
        def task(h: List[str]) -> Any:
            try:
                args = (h,) if label_name is None else (label_name, h)
                return label_fn(*args)
            except Exception as err:
                err_path = "/".join(h)
                msg = f"error occurred ({err}) when getting label of {err_path}"
                print(f"{ERROR_PREFIX}{msg}")
                return None

        if not make_labels_in_parallel:
            return [task(h) for h in tqdm(hierarchy_list)]
        parallel = Parallel(num_jobs, use_tqdm=use_tqdm)
        num_files = len(hierarchy_list)
        random_indices = np.random.permutation(num_files).tolist()
        shuffled = [hierarchy_list[i] for i in random_indices]
        groups = parallel.grouped(task, shuffled).ordered_results
        shuffled_results: List[Any] = sum(groups, [])
        final_results = [None] * num_files
        for idx, rs in zip(random_indices, shuffled_results):
            final_results[idx] = rs
        return final_results

    print(f"{INFO_PREFIX}making labels")
    labels = get_labels(preparation.get_label)
    excluded_indices = {i for i, label in enumerate(labels) if label is None}
    extra_labels_dict: Optional[Dict[str, List[str]]] = None
    extra_labels = preparation.extra_labels
    extra_label_fn = preparation.get_extra_label
    if extra_labels is not None:
        extra_labels_dict = {}
        print(f"{INFO_PREFIX}making extra labels")
        for el_name in extra_labels:
            extra_labels_dict[el_name] = get_labels(extra_label_fn, el_name)
        for extra_labels in extra_labels_dict.values():
            for i, extra_label in enumerate(extra_labels):
                if extra_label is None:
                    excluded_indices.add(i)

    if not to_index:
        labels_dict = {"": labels}
    else:
        label2idx = {label: i for i, label in enumerate(sorted(set(labels)))}
        with open(os.path.join(tgt_folder, f"{LABEL_KEY}2idx.json"), "w") as f:
            json.dump(label2idx, f)
        with open(os.path.join(tgt_folder, f"idx2{LABEL_KEY}.json"), "w") as f:
            json.dump({v: k for k, v in label2idx.items()}, f)
        labels_dict = {"": [label2idx[label] for label in labels]}

    if extra_labels_dict is not None:
        for el_name, label_collection in extra_labels_dict.items():
            if not to_index:
                labels_dict[el_name] = label_collection
            else:
                extra2idx = {
                    extra_label: i
                    for i, extra_label in enumerate(sorted(set(label_collection)))
                }
                with open(os.path.join(tgt_folder, f"{el_name}2idx.json"), "w") as f:
                    json.dump(extra2idx, f)
                with open(os.path.join(tgt_folder, f"idx2{el_name}.json"), "w") as f:
                    json.dump({v: k for k, v in extra2idx.items()}, f)
                labels_dict[el_name] = [extra2idx[el] for el in label_collection]

    # exclude samples
    if excluded_indices:
        print(f"{WARNING_PREFIX}{len(excluded_indices)} samples will be excluded")
    for i in sorted(excluded_indices)[::-1]:
        for sub_labels in labels_dict.values():
            sub_labels.pop(i)
        all_img_paths.pop(i)

    # prepare core
    num_sample = len(all_img_paths)
    if valid_split < 1:
        valid_split = min(max_num_valid, int(round(num_sample * valid_split)))
    assert isinstance(valid_split, int)

    train_portion = (num_sample - valid_split) / num_sample
    label_indices_mapping: Dict[Any, List[int]] = {}
    for i, label in enumerate(labels):
        label_indices_mapping.setdefault(label, []).append(i)
    tuple(map(random.shuffle, label_indices_mapping.values()))
    train_indices_list: List[List[int]] = []
    valid_indices_list: List[List[int]] = []
    for label_indices in label_indices_mapping.values():
        num_label_samples = len(label_indices)
        num_train = int(round(train_portion * num_label_samples))
        num_train = min(num_train, num_label_samples - 1)
        if num_train == 0:
            train_indices_list.append([label_indices[0]])
        else:
            train_indices_list.append(label_indices[:num_train])
        valid_indices_list.append(label_indices[num_train:])

    def propagate(src: List[List[int]], tgt: List[List[int]]) -> None:
        resolved = 0
        src_lengths = list(map(len, src))
        sorted_indices = np.argsort(src_lengths).tolist()[::-1]
        while True:
            for idx in sorted_indices:
                if len(src[idx]) > 1:
                    tgt[idx].append(src[idx].pop())
                resolved += 1
                if resolved == diff:
                    break
            if resolved == diff:
                break

    diff = sum(map(len, valid_indices_list)) - valid_split
    if diff > 0:
        propagate(valid_indices_list, train_indices_list)
    elif diff < 0:
        diff *= -1
        propagate(train_indices_list, valid_indices_list)
    merged_train_indices: List[int] = sorted(set(sum(train_indices_list, [])))
    merged_valid_indices: List[int] = sorted(set(sum(valid_indices_list, [])))
    if train_all_data:
        merged_train_indices.extend(merged_valid_indices)
    train_indices = np.array(merged_train_indices)
    valid_indices = np.array(merged_valid_indices)

    def save(indices: np.ndarray, d_num_jobs: int, dtype: str) -> None:
        def record(idx: int) -> Optional[Tuple[str, Dict[str, Any]]]:
            split_folder = os.path.join(tgt_folder, dtype)
            os.makedirs(split_folder, exist_ok=True)
            img_path = all_img_paths[idx]
            new_img_path = preparation.get_new_img_path(idx, split_folder, img_path)
            try:
                preparation.copy(img_path, new_img_path)
                key = os.path.abspath(new_img_path)
                idx_labels: Dict[str, Any] = {}
                for label_t, t_labels in labels_dict.items():
                    idx_labels[label_t] = {key: t_labels[idx]}
                return key, idx_labels
            except Exception as err:
                print(f"error occurred with {img_path} : {err}")
                return None

        parallel = Parallel(d_num_jobs, use_tqdm=use_tqdm)
        results: List[Tuple[str, Dict[str, Any]]]
        indices = indices.copy()
        np.random.shuffle(indices)
        results = sum(parallel.grouped(record, indices).ordered_results, [])
        d_valid_indices = [i for i, r in enumerate(results) if r is not None]
        results = [results[i] for i in d_valid_indices]
        valid_paths = [all_img_paths[idx] for idx in indices[d_valid_indices]]
        new_paths, all_labels_list = zip(*results)
        merged_labels = shallow_copy_dict(all_labels_list[0])
        for sub_labels_ in all_labels_list[1:]:
            for k, v in shallow_copy_dict(sub_labels_).items():
                merged_labels[k].update(v)
        print(
            "\n".join(
                [
                    "",
                    "=" * 100,
                    f"num {dtype} samples : {len(next(iter(merged_labels.values())))}",
                    f"num {dtype} label types : {len(merged_labels)}",
                    "-" * 100,
                    "",
                ]
            )
        )
        dtype_folder = os.path.join(tgt_folder, dtype)
        with open(os.path.join(dtype_folder, "paths.json"), "w") as f_:
            json.dump(new_paths, f_)
        path_mapping = dict(zip(new_paths, valid_paths))
        with open(os.path.join(dtype_folder, "path_mapping.json"), "w") as f_:
            json.dump(path_mapping, f_)
        for label_type, type_labels in merged_labels.items():
            delim = "_" if label_type else ""
            label_file = f"{label_type}{delim}{LABEL_KEY}.json"
            with open(os.path.join(dtype_folder, label_file), "w") as f_:
                json.dump(type_labels, f_)
        # lmdb
        if lmdb_config is None:
            return None
        local_lmdb_config = shallow_copy_dict(lmdb_config)
        local_lmdb_config.setdefault("path", _default_lmdb_path(tgt_folder, dtype))
        local_lmdb_config.setdefault("map_size", 1099511627776 * 2)
        db = lmdb.open(**local_lmdb_config)
        context = db.begin(write=True)
        d_num_samples = len(results)
        iterator = zip(range(d_num_samples), new_paths, all_labels_list)
        if use_tqdm:
            iterator = tqdm(iterator, total=d_num_samples, desc="lmdb")
        for i, path, i_labels in iterator:
            i_new_labels = {}
            for k, v in i_labels.items():
                vv = v[path]
                if isinstance(vv, str):
                    if vv.endswith(".npy"):
                        vv = np.load(vv)
                i_new_labels[k] = vv
            context.put(
                str(i).encode("ascii"),
                dill.dumps(LMDBItem(np.array(Image.open(path)), i_new_labels)),
            )
        context.put(
            "length".encode("ascii"),
            str(d_num_samples).encode("ascii"),
        )
        context.commit()
        db.sync()
        db.close()

    save(train_indices, max(1, num_jobs), "train")
    save(valid_indices, max(1, num_jobs // 2), "valid")
    return tgt_folder


def prepare_image_folder_data(
    src_folder: str,
    tgt_folder: str,
    *,
    to_index: bool,
    batch_size: int,
    prefix: Optional[str] = None,
    preparation: _PreparationProtocol = DefaultPreparation(),
    num_workers: int = 0,
    shuffle: bool = True,
    prefetch_device: Optional[Union[int, str]] = None,
    pin_memory_device: Optional[Union[int, str]] = None,
    transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
    transform_config: Optional[Dict[str, Any]] = None,
    test_shuffle: Optional[bool] = None,
    test_transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
    test_transform_config: Optional[Dict[str, Any]] = None,
    train_all_data: bool = False,
    force_rerun: bool = False,
    extensions: Optional[Set[str]] = None,
    make_labels_in_parallel: bool = False,
    num_jobs: int = 8,
    valid_split: Union[int, float] = 0.1,
    max_num_valid: int = 10000,
    lmdb_config: Optional[Dict[str, Any]] = None,
    use_tqdm: bool = True,
) -> PrepareResults:
    tgt_folder = prepare_image_folder(
        src_folder,
        tgt_folder,
        to_index=to_index,
        prefix=prefix,
        preparation=preparation,
        force_rerun=force_rerun,
        extensions=extensions,
        make_labels_in_parallel=make_labels_in_parallel,
        num_jobs=num_jobs,
        train_all_data=train_all_data,
        valid_split=valid_split,
        max_num_valid=max_num_valid,
        lmdb_config=lmdb_config,
        use_tqdm=use_tqdm,
    )
    data = ImageFolderData(
        tgt_folder,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        prefetch_device=prefetch_device,
        pin_memory_device=pin_memory_device,
        extra_label_names=preparation.extra_labels,
        transform=transform,
        transform_config=transform_config,
        test_shuffle=test_shuffle,
        test_transform=test_transform,
        test_transform_config=test_transform_config,
        lmdb_config=lmdb_config,
    )
    return PrepareResults(data, tgt_folder)


__all__ = [
    "TensorData",
    "DummyData",
    "MLData",
    "MLInferenceData",
    "ImageFolderData",
    "InferenceImageFolderData",
    "DefaultPreparation",
    "prepare_image_folder",
    "prepare_image_folder_data",
]
