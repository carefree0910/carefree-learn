import os
import dill
import json
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
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.dist import Parallel
from cftool.misc import walk
from cftool.misc import print_info
from cftool.misc import print_error
from cftool.misc import print_warning
from cftool.misc import get_arguments
from cftool.misc import shallow_copy_dict
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type
from torch.utils.data import Dataset

from .cv import default_lmdb_path
from .cv import CVLoader
from .cv import LMDBItem
from .cv import CVDataset
from .cv import Transforms
from .cv import ImageFolderDataset
from .cv import InferenceImageFolderDataset
from .ml import MLLoader
from .ml import MLDataset
from .ml import IMLData
from .ml import IMLDataInfo
from .ml import IMLDataModifier
from .core import DLLoader
from .core import DLDataset
from .core import DataLoader
from .core import DLDataModule
from ..types import data_type
from ..types import sample_weights_type
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..constants import ORIGINAL_LABEL_KEY

try:
    import lmdb
except:
    lmdb = None
try:
    from cfdata.tabular.api import TabularData
except:
    TabularData = None


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
        item = {INPUT_KEY: self.x[index]}
        if self.y is not None:
            label = self.y[index]
            item.update({LABEL_KEY: label, ORIGINAL_LABEL_KEY: label})
        if self.others is not None:
            for k, v in self.others.items():
                item[k] = v[index]
        return item

    def __len__(self) -> int:
        return self.x.shape[0]


class TensorDictDataset(Dataset):
    def __init__(
        self,
        x: tensor_dict_type,
        y: Optional[Tensor],
    ):
        self.x = x
        self.y = y
        self.x_keys = sorted(self.x)

    def __getitem__(self, index: int) -> tensor_dict_type:
        item = {k: self.x[k][index] for k in self.x_keys}
        if self.y is not None:
            label = self.y[index]
            item.update({LABEL_KEY: label, ORIGINAL_LABEL_KEY: label})
        return item

    def __len__(self) -> int:
        return self.x[self.x_keys[0]].shape[0]


@DLDataModule.register("tensor")
class TensorData(DLDataModule):
    files = [
        "x_train.pt",
        "y_train.pt",
        "x_valid.pt",
        "y_valid.pt",
        "train_others.pt",
        "valid_others.pt",
    ]

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

    def _save_data(self, data_folder: str) -> None:
        all_data = [self.x_train, self.y_train, self.x_valid, self.y_valid]
        all_data += [self.train_others, self.valid_others]
        for data, file in zip(all_data, self.files):
            if data is not None:
                torch.save(data, os.path.join(data_folder, file))

    @classmethod
    def _load(
        cls,
        data_folder: str,
        info: Dict[str, Any],
        sample_weights: sample_weights_type,
    ) -> "TensorData":
        args = []
        for file in cls.files:
            path = os.path.join(data_folder, file)
            args.append(None if not os.path.isfile(path) else torch.load(path))
        data = cls(*args, **info)
        data.prepare(sample_weights)
        return data


@DLDataModule.register("tensor_dict")
class TensorDictData(DLDataModule):
    files = ["x_train.pt", "y_train.pt", "x_valid.pt", "y_valid.pt"]

    def __init__(
        self,
        x_train: tensor_dict_type,
        y_train: Optional[Tensor] = None,
        x_valid: Optional[tensor_dict_type] = None,
        y_valid: Optional[Tensor] = None,
        *,
        shuffle: bool = True,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    @property
    def info(self) -> Dict[str, Any]:
        return self.kw

    # TODO : support sample weights
    def prepare(self, sample_weights: sample_weights_type) -> None:
        def _get_data(x: Any, y: Any) -> DLDataset:
            return DLDataset(TensorDictDataset(x, y))

        self.train_data = _get_data(self.x_train, self.y_train)
        if self.x_valid is None:
            self.valid_data = None
        else:
            self.valid_data = _get_data(self.x_valid, self.y_valid)

    def initialize(self) -> Tuple[DLLoader, Optional[DLLoader]]:
        train_loader = DLLoader(DataLoader(self.train_data, **self.kw))  # type: ignore
        if self.valid_data is None:
            valid_loader = None
        else:
            valid_loader = DLLoader(DataLoader(self.valid_data, **self.kw))  # type: ignore
        return train_loader, valid_loader

    def _save_data(self, data_folder: str) -> None:
        all_data = [self.x_train, self.y_train, self.x_valid, self.y_valid]
        for data, file in zip(all_data, self.files):
            if data is not None:
                torch.save(data, os.path.join(data_folder, file))

    @classmethod
    def _load(
        cls,
        data_folder: str,
        info: Dict[str, Any],
        sample_weights: sample_weights_type,
    ) -> "TensorData":
        args = []
        for file in cls.files:
            path = os.path.join(data_folder, file)
            args.append(None if not os.path.isfile(path) else torch.load(path))
        data = cls(*args, **info)  # type: ignore
        data.prepare(sample_weights)
        return data


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


@IMLDataModifier.register("_internal.ml")
class _InternalMLDataModifier(IMLDataModifier):
    data: "MLData"

    def save(self, data_folder: str) -> None:
        with open(os.path.join(data_folder, self.data.arguments_file), "w") as f:
            json.dump(self.data.arguments, f)
        all_data = [
            self.data.x_train,
            self.data.y_train,
            self.data.x_valid,
            self.data.y_valid,
        ]
        for data, file in zip(all_data, self.data.data_files):
            if data is not None:
                np.save(os.path.join(data_folder, file), data)
        if self.data.train_others is not None:
            for k, v in self.data.train_others.items():
                np.save(os.path.join(data_folder, f"{k}_train.npy"), v)
        if self.data.valid_others is not None:
            for k, v in self.data.valid_others.items():
                np.save(os.path.join(data_folder, f"{k}_valid.npy"), v)
        if self.data.cf_data is not None:
            full_cf_data_folder = os.path.join(data_folder, self.data.full_cf_data_name)
            self.data.cf_data.save(full_cf_data_folder, retain_data=True)

    @classmethod
    def load(cls, data_folder: str, info: Dict[str, Any]) -> "MLData":
        args = []
        for file in MLData.data_files:
            path = os.path.join(data_folder, file)
            args.append(None if not os.path.isfile(path) else np.load(path))
        with open(os.path.join(data_folder, MLData.arguments_file), "r") as f:
            kwargs = json.load(f)
        train_others = {}
        valid_others = {}
        for file in os.listdir(data_folder):
            if file in MLData.data_files:
                continue
            path = os.path.join(data_folder, file)
            if file.endswith("_train"):
                train_others[file.split("_train")[0]] = np.load(path)
            elif file.endswith("_valid"):
                valid_others[file.split("_valid")[0]] = np.load(path)
        if train_others:
            kwargs["train_others"] = train_others
        if valid_others:
            kwargs["valid_others"] = valid_others
        if info["cf_data"] is not None:
            if TabularData is None:
                raise ValueError(
                    "`carefree-data` needs to be installed "
                    "to load `MLData` with `cf_data` defined"
                )
            full_cf_data_folder = os.path.join(data_folder, MLData.full_cf_data_name)
            kwargs["cf_data"] = TabularData.load(full_cf_data_folder)
        return MLData(*args, **kwargs)

    def permute_info_in_save(self, info: Dict[str, Any]) -> None:
        if info["cf_data"] is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_name = os.path.join(tmp_dir, self.data.tmp_cf_data_name)
                info["cf_data"].save(tmp_name, retain_data=False)
                zip_file = f"{tmp_name}.zip"
                with open(zip_file, "rb") as f:
                    info["cf_data"] = f.read()
                os.remove(zip_file)

    @classmethod
    def permute_info_in_load(
        cls,
        data_cls: Type["MLData"],
        info: Dict[str, Any],
    ) -> None:
        cf_data = info["cf_data"]
        if cf_data is None:
            return
        if TabularData is None:
            raise ValueError(
                "`carefree-data` needs to be installed "
                "to load `MLData` with `cf_data` defined"
            )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_name = os.path.join(tmp_dir, data_cls.tmp_cf_data_name)
            zip_file = f"{tmp_name}.zip"
            with open(zip_file, "wb") as f:
                f.write(cf_data)
            info["cf_data"] = TabularData.load(tmp_name)
            os.remove(zip_file)


@DLDataModule.register("ml")
class MLData(IMLData):
    modifier = "_internal.ml"

    cf_data: Optional[TabularData]
    train_data: MLDataset
    valid_data: Optional[MLDataset]
    train_cf_data: Optional[TabularData]
    valid_cf_data: Optional[TabularData]

    data_files = [
        "x_train.npy",
        "y_train.npy",
        "x_valid.npy",
        "y_valid.npy",
    ]
    arguments_file = "arguments.json"
    tmp_cf_data_name = ".tmp_cf_data"
    full_cf_data_name = "cf_data"

    def __init__(
        self,
        x_train: data_type,
        y_train: data_type = None,
        x_valid: data_type = None,
        y_valid: data_type = None,
        *,
        train_others: Optional[np_dict_type] = None,
        valid_others: Optional[np_dict_type] = None,
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
        contains_labels: bool = True,
    ):
        self.arguments = shallow_copy_dict(get_arguments())
        pop_keys = [
            "x_train",
            "y_train",
            "x_valid",
            "y_valid",
            "train_others",
            "valid_others",
            "cf_data",
        ]
        for key in pop_keys:
            self.arguments.pop(key)
        assert x_train is not None
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.train_others = train_others
        self.valid_others = valid_others
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
        self.for_inference = for_inference
        self.contains_labels = contains_labels
        self.loaded = False

    def get_info(self) -> IMLDataInfo:
        return IMLDataInfo(
            self.input_dim,
            self.num_history,
            self.num_classes,
            self.is_classification,
        )

    @property
    def info(self) -> Dict[str, Any]:
        info = super().info
        info["cf_data"] = self.cf_data
        return info

    def prepare(self, sample_weights: sample_weights_type) -> None:
        train_others = self.train_others or {}
        valid_others = self.valid_others or {}
        self.train_weights, self.valid_weights = _split_sw(sample_weights)
        if self.cf_data is not None:
            if self.for_inference:
                train_xy = self.cf_data.transform(
                    self.x_train,
                    None,
                    contains_labels=self.contains_labels,
                ).xy
                self.train_data = MLDataset(*train_xy)
                self.valid_data = None
                self.train_cf_data = self.cf_data
                self.valid_cf_data = None
                # if `for_inference` is True, these properties are not needed
                self.input_dim = self.num_classes = self.is_classification = None
            else:
                if not self.loaded:
                    self.cf_data.read(self.x_train, self.y_train, **self.read_config)
                if self.x_valid is not None:
                    self.train_cf_data = self.cf_data
                    self.valid_cf_data = self.cf_data.copy_to(
                        self.x_valid, self.y_valid
                    )
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
                            max_split = int(
                                round(num_data * self.max_valid_split_ratio)
                            )
                            max_split = min(max_split, self.max_valid_split)
                            split = min(num_split, max_split)
                    if split <= 0:
                        self.train_cf_data = self.cf_data
                        self.valid_cf_data = None
                    else:
                        rs = self.cf_data.split(split, order=self.valid_split_order)
                        self.train_cf_data = rs.remained
                        self.valid_cf_data = rs.split
                train_xy = self.train_cf_data.processed.xy
                self.train_data = MLDataset(*train_xy, **train_others)
                if self.valid_cf_data is None:
                    self.valid_data = None
                else:
                    valid_xy = self.valid_cf_data.processed.xy
                    self.valid_data = MLDataset(*valid_xy, **valid_others)
                # initialize properties with train_cf_data
                self.input_dim = self.train_cf_data.processed_dim
                self.num_classes = self.train_cf_data.num_classes
                if self.is_classification is None:
                    self.is_classification = self.train_cf_data.is_clf
            return None
        if isinstance(self.x_train, str):
            raise ValueError("`cf_data` should be provided when `x_train` is `str`")
        self.num_classes = None
        self.input_dim = self.x_train.shape[-1]
        self.train_data = MLDataset(self.x_train, self.y_train, **train_others)
        if self.x_valid is None or self.y_valid is None:
            self.valid_data = None
        else:
            self.valid_data = MLDataset(self.x_valid, self.y_valid, **valid_others)

    def initialize(self) -> Tuple[MLLoader, Optional[MLLoader]]:
        train_loader = MLLoader(
            self.train_data,
            name=None if self.for_inference else "train",
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

    @classmethod
    def with_cf_data(
        cls,
        *args: Any,
        is_classification: Optional[bool] = None,
        cf_data_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "MLData":
        if TabularData is None:
            raise ValueError("`carefree-data` needs to be installed for `with_cf_data`")
        if cf_data_config is None:
            cf_data_config = {}
        cf_data_config["default_categorical_process"] = "identical"
        if is_classification is not None:
            cf_data_config["task_type"] = "clf" if is_classification else "reg"
        kwargs["is_classification"] = is_classification
        kwargs["cf_data"] = TabularData(**(cf_data_config or {}))
        return cls(*args, **kwargs)


class MLInferenceData(MLData):
    def __init__(
        self,
        x: data_type,
        y: data_type = None,
        *,
        shuffle: bool = False,
        contains_labels: bool = True,
        cf_data: Optional[TabularData] = None,
    ):
        super().__init__(
            x,
            y,
            cf_data=cf_data,
            shuffle_train=shuffle,
            for_inference=True,
            contains_labels=contains_labels,
        )
        self.prepare(None)


# cv


class CVDataModule(DLDataModule, metaclass=ABCMeta):
    arguments: Dict[str, Any]
    test_transform: Optional[Transforms]
    transform_file = "transform.pkl"
    arguments_file = "arguments.json"

    def _save_info(self, folder: str) -> None:
        super()._save_info(folder)
        with open(os.path.join(folder, self.transform_file), "wb") as f:
            dill.dump(self.test_transform, f)

    @classmethod
    def _load_info(cls, folder: str) -> Dict[str, Any]:
        info = super()._load_info(folder)
        transform_path = os.path.join(folder, cls.transform_file)
        if not os.path.isfile(transform_path):
            test_transform = None
        else:
            with open(transform_path, "rb") as f:
                test_transform = dill.load(f)
        info["test_transform"] = test_transform
        return info

    def _save_data(self, data_folder: str) -> None:
        with open(os.path.join(data_folder, self.arguments_file), "w") as f:
            json.dump(self.arguments, f)

    @classmethod
    def _load(
        cls,
        data_folder: str,
        info: Dict[str, Any],
        sample_weights: sample_weights_type,
    ) -> "CVDataModule":
        with open(os.path.join(data_folder, cls.arguments_file), "r") as f:
            kwargs = json.load(f)
        data = cls(**kwargs)
        data.test_transform = info["test_transform"]
        data.prepare(sample_weights)
        return data


@DLDataModule.register("image_folder")
class ImageFolderData(CVDataModule):
    def __init__(
        self,
        folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_train_last: bool = True,
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
        self.arguments = shallow_copy_dict(get_arguments())
        self.folder = folder
        self.shuffle = shuffle
        self.drop_train_last = drop_train_last
        if not torch.cuda.is_available():
            fmt = "cuda is not available but {} is provided, which will have no effect"
            if prefetch_device is not None:
                print_warning(f"{fmt.format('`prefetch_device`')}")
                prefetch_device = None
            if pin_memory_device is not None:
                print_warning(f"{fmt.format('`pin_memory_device`')}")
                pin_memory_device = None
        if prefetch_device is not None:
            if pin_memory_device is None:
                pin_memory_device = prefetch_device
            if pin_memory_device is not None:
                if str(prefetch_device) != str(pin_memory_device):
                    print_warning(
                        "`prefetch_device` and `pin_memory_device` "
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
        use_train_as_valid = not os.path.isdir(os.path.join(self.folder, "valid"))
        self.valid_data = CVDataset(
            ImageFolderDataset(
                self.folder,
                "train" if use_train_as_valid else "valid",
                self.test_transform,
                extra_label_names=self.extra_label_names,
                lmdb_config=self.lmdb_config,
            )
        )

    def initialize(self) -> Tuple[CVLoader, Optional[CVLoader]]:
        if self.pin_memory_device is not None:
            torch.cuda.set_device(self.pin_memory_device)
        d = shallow_copy_dict(self.kw)
        trd = shallow_copy_dict(d)
        trd["drop_last"] = self.drop_train_last
        kw = {"prefetch_device": self.prefetch_device}
        train_loader = CVLoader(DataLoader(self.train_data, **trd), **kw)  # type: ignore
        d["shuffle"] = self.test_shuffle or self.shuffle
        valid_loader = CVLoader(DataLoader(self.valid_data, **d), **kw)  # type: ignore
        return train_loader, valid_loader

    @staticmethod
    def switch_prefix(src: str, previous: str, now: str) -> None:
        for split in ["train", "valid"]:
            split_folder = os.path.join(src, split)
            for file in os.listdir(split_folder):
                if not file.endswith(".json"):
                    continue
                path = os.path.join(split_folder, file)
                print(f"switching prefix for '{path}'")
                with open(path, "r") as f:
                    rs = json.load(f)
                new_rs: Union[List, Dict[str, Any]]
                if isinstance(rs, list):
                    new_rs = [item.replace(previous, now) for item in rs]
                else:
                    new_rs = {}
                    for k, v in rs.items():
                        new_rs[k.replace(previous, now)] = v.replace(previous, now)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(new_rs, f, ensure_ascii=False)


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
        self.arguments = shallow_copy_dict(get_arguments())
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

    def is_ready(self, tgt_folder: str, valid_split: Union[int, float]) -> bool:
        candidates = ["train"]
        if valid_split > 0:
            candidates.append("valid")
        for split in candidates:
            extra_keys = [f"{key}_labels" for key in self.extra_labels or []]
            for key in [LABEL_KEY] + extra_keys:
                path = os.path.join(tgt_folder, split, f"{key}.json")
                if not os.path.isfile(path):
                    return False
        return True

    def get_num_classes(self, tgt_folder: str) -> Dict[str, int]:
        num_classes = {}
        for label_name in [LABEL_KEY] + (self.extra_labels or []):
            path = os.path.join(tgt_folder, f"idx2{label_name}.json")
            if not os.path.isfile(path):
                num_classes[label_name] = 0
                continue
            with open(path, "r") as f:
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

    if not force_rerun and preparation.is_ready(tgt_folder, valid_split):
        return tgt_folder

    preparation.prepare_src_folder(src_folder)
    if os.path.isdir(tgt_folder):
        print_warning(f"'{tgt_folder}' already exists, it will be removed")
        shutil.rmtree(tgt_folder)
    os.makedirs(tgt_folder, exist_ok=True)

    print_info("collecting hierarchies")

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
                print_error(f"error occurred ({err}) when getting label of {err_path}")
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

    print_info("making labels")
    labels = get_labels(preparation.get_label)
    excluded_indices = {i for i, label in enumerate(labels) if label is None}
    extra_labels_dict: Optional[Dict[str, List[str]]] = None
    extra_labels = preparation.extra_labels
    extra_label_fn = preparation.get_extra_label
    if extra_labels is not None:
        extra_labels_dict = {}
        print_info("making extra labels")
        for el_name in extra_labels:
            extra_labels_dict[el_name] = get_labels(extra_label_fn, el_name)
        for extra_labels in extra_labels_dict.values():
            for i, extra_label in enumerate(extra_labels):
                if extra_label is None:
                    excluded_indices.add(i)

    def get_raw_2idx(raw_labels: List[Any]) -> Dict[Any, Any]:
        return {
            v: numpy_token if isinstance(v, str) and v.endswith(".npy") else v
            for v in raw_labels
        }

    def check_dump_mappings(l2i: Dict[Any, Any]) -> bool:
        all_indices = set(l2i.values())
        if len(all_indices) > 1:
            return True
        return list(all_indices)[0] != numpy_token

    numpy_token = "[NUMPY]"
    if to_index:
        label2idx = {label: i for i, label in enumerate(sorted(set(labels)))}
        labels_dict = {"": [label2idx[label] for label in labels]}
        dump_mappings = True
    else:
        labels_dict = {"": labels}
        label2idx = get_raw_2idx(sorted(set(labels)))
        dump_mappings = check_dump_mappings(label2idx)

    open_file_from = lambda folder: lambda file: open(
        os.path.join(folder, file), "w", encoding="utf-8"
    )
    open_tgt_file = open_file_from(tgt_folder)

    if dump_mappings:
        with open_tgt_file(f"{LABEL_KEY}2idx.json") as f:
            json.dump(label2idx, f, ensure_ascii=False)
        with open_tgt_file(f"idx2{LABEL_KEY}.json") as f:
            json.dump({v: k for k, v in label2idx.items()}, f, ensure_ascii=False)

    if extra_labels_dict is not None:
        for el_name, label_collection in extra_labels_dict.items():
            if not to_index:
                labels_dict[el_name] = label_collection  # type: ignore
                extra2idx = get_raw_2idx(sorted(set(label_collection)))
                dump_mappings = check_dump_mappings(extra2idx)
            else:
                extra2idx = {
                    extra_label: i  # type: ignore
                    for i, extra_label in enumerate(sorted(set(label_collection)))
                }
                labels_dict[el_name] = [extra2idx[el] for el in label_collection]
                dump_mappings = True
            if dump_mappings:
                with open_tgt_file(f"{el_name}2idx.json") as f:
                    json.dump(extra2idx, f, ensure_ascii=False)
                with open_tgt_file(f"idx2{el_name}.json") as f:
                    eld = {v: k for k, v in extra2idx.items()}
                    json.dump(eld, f, ensure_ascii=False)

    # exclude samples
    if excluded_indices:
        print_warning(f"{len(excluded_indices)} samples will be excluded")
    for i in sorted(excluded_indices)[::-1]:
        for sub_labels in labels_dict.values():
            sub_labels.pop(i)
        all_img_paths.pop(i)

    # prepare core
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
        open_dtype_file = open_file_from(os.path.join(tgt_folder, dtype))

        with open_dtype_file("paths.json") as f_:
            json.dump(new_paths, f_, ensure_ascii=False)
        path_mapping = dict(zip(new_paths, valid_paths))
        with open_dtype_file("path_mapping.json") as f_:
            json.dump(path_mapping, f_, ensure_ascii=False)
        for label_type, type_labels in merged_labels.items():
            delim = "_" if label_type else ""
            label_file = f"{label_type}{delim}{LABEL_KEY}.json"
            with open_dtype_file(label_file) as f_:
                json.dump(type_labels, f_, ensure_ascii=False)
        # lmdb
        if lmdb_config is None or lmdb is None:
            if lmdb_config is not None:
                print_warning(
                    "`lmdb` is not installed, so `lmdb_config` will be ignored"
                )
            return None
        local_lmdb_config = shallow_copy_dict(lmdb_config)
        local_lmdb_config.setdefault("path", default_lmdb_path(tgt_folder, dtype))
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

    num_sample = len(all_img_paths)
    if valid_split < 1:
        valid_split = min(max_num_valid, int(round(num_sample * valid_split)))
    assert isinstance(valid_split, int)

    if valid_split <= 0:
        save(np.arange(num_sample), max(1, num_jobs), "train")
        return tgt_folder

    train_portion = (num_sample - valid_split) / num_sample
    label_indices_mapping: Dict[Any, List[int]] = {}
    for i, label in enumerate(labels):
        if isinstance(label, str) and label.endswith(".npy"):
            label = numpy_token
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
    drop_train_last: bool = True,
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
        drop_train_last=drop_train_last,
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
    "TensorDictData",
    "DummyData",
    "MLData",
    "MLInferenceData",
    "CVDataModule",
    "ImageFolderData",
    "InferenceImageFolderData",
    "DefaultPreparation",
    "prepare_image_folder",
    "prepare_image_folder_data",
]
