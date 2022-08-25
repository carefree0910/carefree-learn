import os
import dill
import json
import torch
import random
import shutil

import numpy as np

from abc import abstractmethod
from abc import ABCMeta
from PIL import Image
from tqdm import tqdm
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
from cftool.misc import is_numeric
from cftool.misc import print_info
from cftool.misc import print_error
from cftool.misc import print_warning
from cftool.misc import shallow_copy_dict
from cftool.misc import WithRegister
from cftool.array import to_torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

from .core import DLLoader
from .core import DLDataset
from .core import DataLoader
from .core import DLDataModule
from ..types import sample_weights_type
from ..protocol import IDataset
from ..protocol import IDataLoader
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..misc.toolkit import ConfigMeta

try:
    import lmdb
except:
    lmdb = None


@IDataset.register("cv")
class CVDataset(DLDataset):
    pass


@IDataLoader.register("cv")
class CVLoader(DLLoader):
    data: CVDataset


cf_transforms: Dict[str, Type["Transforms"]] = {}


class Transforms(WithRegister["Transforms"]):
    d = cf_transforms

    fn: Any

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def __call__(self, inp: Any, *args: Any, **kwargs: Any) -> Any:
        if self.need_batch_process and not isinstance(inp, dict):
            raise ValueError(f"`inp` should be a batch for {self.__class__.__name__}")
        return self.fn(inp, *args, **kwargs)

    @property
    @abstractmethod
    def need_batch_process(self) -> bool:
        pass

    @property
    def need_numpy(self) -> bool:
        return self.need_batch_process

    @classmethod
    def convert(
        cls,
        transform: Optional[Union[str, List[str], "Transforms", Callable]],
        transform_config: Optional[Dict[str, Any]] = None,
    ) -> Optional["Transforms"]:
        if transform is None:
            return None
        if isinstance(transform, Transforms):
            return transform
        if callable(transform):
            need_batch = (transform_config or {}).get("need_batch_process", False)
            return Function(transform, need_batch)
        if transform_config is None:
            transform_config = {}
        if isinstance(transform, str):
            return cls.make(transform, transform_config)
        transform_list = [cls.make(t, transform_config.get(t, {})) for t in transform]
        return Compose(transform_list)

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> "Transforms":
        split = name.split("_")
        if len(split) >= 3 and split[-2] == "with":
            name = "_".join(split[:-2])
            config.setdefault("label_alias", split[-1])
        return super().make(name, config)


class Function(Transforms):
    def __init__(self, fn: Callable, need_batch_process: bool = False):
        super().__init__()
        self.fn = fn
        self._need_batch_process = need_batch_process

    @property
    def need_batch_process(self) -> bool:
        return self._need_batch_process


@Transforms.register("compose")
class Compose(Transforms):
    def __init__(self, transform_list: List[Transforms]):
        super().__init__()
        if len(set(t.need_batch_process for t in transform_list)) > 1:
            raise ValueError(
                "all transforms should have identical "
                "`need_batch_process` property in `Compose`"
            )
        self.fn = transforms.Compose(transform_list)
        self.transform_list = transform_list

    @property
    def need_batch_process(self) -> bool:
        return self.transform_list[0].need_batch_process

    @property
    def need_numpy(self) -> bool:
        return self.transform_list[0].need_numpy


def default_lmdb_path(folder: str, split: str) -> str:
    return f"{folder}.{split}"


class LMDBItem(NamedTuple):
    image: np.ndarray
    labels: Dict[str, Any]


class ImageDatasetMixin(Dataset):
    context: Optional[Any] = None
    labels: Optional[Dict[str, Any]] = None

    img_paths: List[str]
    transform: Optional[Transforms]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.context is not None:
            key = str(index).encode("ascii")
            item: LMDBItem = dill.loads(self.context.get(key))
            img = Image.fromarray(item.image)
            labels = item.labels
            for k, v in labels.items():
                if is_numeric(v):
                    labels[k] = np.array([v])
        else:
            file = self.img_paths[index]
            img = Image.open(file)
            if self.labels is None:
                labels = None
            else:
                labels = {k: v[file] for k, v in self.labels.items()}
                for k, v in labels.items():
                    if is_numeric(v):
                        v = np.array([v])
                    elif isinstance(v, str) and v.endswith(".npy"):
                        v = np.load(v)
                    labels[k] = v
        if self.transform is None:
            img = to_torch(np.array(img).astype(np.float32))
            sample = {INPUT_KEY: img}
            if labels is not None:
                for k, v in labels.items():
                    if isinstance(v, np.ndarray):
                        labels[k] = to_torch(v)
                sample.update(labels)
            return sample
        if not self.transform.need_numpy:
            sample = {INPUT_KEY: img}
        else:
            sample = {INPUT_KEY: np.array(img).astype(np.float32) / 255.0}
        if self.transform.need_batch_process:
            if labels is not None:
                sample.update(labels)
            return self.transform(sample)
        sample = {INPUT_KEY: self.transform(sample[INPUT_KEY])}
        if labels is not None:
            for k, v in labels.items():
                if isinstance(v, np.ndarray):
                    labels[k] = to_torch(v)
            sample.update(labels)
        return sample


class ImageFolderDataset(ImageDatasetMixin):
    def __init__(
        self,
        folder: str,
        split: str,
        transform: Optional[Union[str, List[str], Transforms, Callable]],
        transform_config: Optional[Dict[str, Any]] = None,
        extra_label_names: Optional[List[str]] = None,
        lmdb_config: Optional[Dict[str, Any]] = None,
    ):
        self.folder = os.path.abspath(os.path.join(folder, split))
        support_appendix = {".jpg", ".png", ".npy"}
        self.img_paths = list(
            map(
                lambda file: os.path.join(self.folder, file),
                filter(
                    lambda file: file[-4:] in support_appendix,
                    os.listdir(self.folder),
                ),
            )
        )
        if lmdb_config is None or lmdb is None:
            if lmdb_config is not None:
                print_warning(
                    "`lmdb` is not installed, so `lmdb_config` will be ignored"
                )
            self.lmdb = self.context = None
            with open(os.path.join(self.folder, f"{LABEL_KEY}.json"), "r") as f:
                self.labels = {LABEL_KEY: json.load(f)}
            if extra_label_names is not None:
                for name in extra_label_names:
                    el_file = f"{name}_{LABEL_KEY}.json"
                    with open(os.path.join(self.folder, el_file), "r") as f:
                        self.labels[name] = json.load(f)
            self.length = len(self.img_paths)
        else:
            self.lmdb_config = shallow_copy_dict(lmdb_config)
            self.lmdb_config.setdefault("path", default_lmdb_path(folder, split))
            self.lmdb_config.setdefault("lock", False)
            self.lmdb_config.setdefault("meminit", False)
            self.lmdb_config.setdefault("readonly", True)
            self.lmdb_config.setdefault("readahead", False)
            self.lmdb_config.setdefault("max_readers", 32)
            self.lmdb = lmdb.open(**self.lmdb_config)
            self.context = self.lmdb.begin(buffers=True, write=False)
            with self.lmdb.begin(write=False) as context:
                self.length = int(context.get("length".encode("ascii")).decode("ascii"))
        self.transform = Transforms.convert(transform, transform_config)

    def __len__(self) -> int:
        return self.length

    def get_sample_weights_with(
        self,
        label_mapping: Dict[str, Any],
        label_weights: Dict[str, float],
    ) -> List[float]:
        try:
            labels = [label_mapping[path] for path in self.img_paths]
        except:
            missing_keys = [p for p in self.img_paths if p not in label_mapping]
            print_error(
                "`label_mapping` does not cover the entire dataset, "
                f"{len(missing_keys)} keys are missing: ({missing_keys[0]}, ...)"
            )
            raise
        try:
            return [label_weights[l] for l in labels]
        except:
            missing_keys = [l for l in labels if l not in label_weights]
            print_error(
                "`label_weights` does not cover the entire dataset, "
                f"{len(missing_keys)} labels are missing: ({missing_keys[0]}, ...)"
            )
            raise


class InferenceImagePathsDataset(ImageDatasetMixin):
    def __init__(self, paths: List[str], transform: Optional[Transforms]):
        self.img_paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def make_loader(
        self,
        batch_size: int,
        num_workers: int = 0,
        prefetch_device: Optional[Union[int, str]] = None,
    ) -> CVLoader:
        return CVLoader(
            DataLoader(self, batch_size, num_workers=num_workers),
            prefetch_device=prefetch_device,
        )


class InferenceImageFolderDataset(InferenceImagePathsDataset):
    def __init__(self, folder: str, transform: Optional[Transforms]):
        self.folder = os.path.abspath(folder)
        img_paths: List[str] = []
        walk(
            self.folder,
            lambda _, path: img_paths.append(path),
            {".jpg", ".png"},
        )
        super().__init__(img_paths, transform)


# api


class CVDataModule(DLDataModule, metaclass=ABCMeta):
    config: Dict[str, Any]
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
            json.dump(self.config, f)

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
class ImageFolderData(CVDataModule, metaclass=ConfigMeta):
    train_data: CVDataset
    valid_data: CVDataset

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
        label_mapping: Optional[Dict[str, Any]] = None,
        label_weights: Optional[Dict[str, float]] = None,
        lmdb_config: Optional[Dict[str, Any]] = None,
    ):
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
        self.label_mapping = label_mapping
        self.label_weights = label_weights
        self.lmdb_config = lmdb_config
        self.kw = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory_device is not None,
        )
        self.train_weights = None

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

    @property
    def use_sw(self) -> bool:
        return self.label_mapping is not None and self.label_weights is not None

    def setup_train_weights(self) -> None:
        if self.use_sw:
            self.train_weights = self.train_data.dataset.get_sample_weights_with(
                self.label_mapping,
                self.label_weights,
            )

    def prepare(self, sample_weights: sample_weights_type) -> None:
        if sample_weights is not None:
            print_warning(
                "`sample_weights` will not take effect in `ImageFolderData`, "
                "please use `label_mappings` and `label_weights` instead"
            )
        self.train_data = CVDataset(
            ImageFolderDataset(
                self.folder,
                "train",
                self.transform,
                extra_label_names=self.extra_label_names,
                lmdb_config=self.lmdb_config,
            )
        )
        self.setup_train_weights()
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
        if self.train_weights is not None:
            trd.pop("shuffle")
            trd["sampler"] = self.make_weighted_sampler(self.train_weights)
        kw = {"prefetch_device": self.prefetch_device}
        train_loader = CVLoader(DataLoader(self.train_data, **trd), **kw)  # type: ignore
        if self.test_shuffle is not None:
            d["shuffle"] = self.test_shuffle
        valid_loader = CVLoader(DataLoader(self.valid_data, **d), **kw)  # type: ignore
        return train_loader, valid_loader

    @staticmethod
    def make_weighted_sampler(sample_weights: List[float]) -> WeightedRandomSampler:
        weights = to_torch(np.array(sample_weights, np.float32))
        return WeightedRandomSampler(weights, len(weights), replacement=True)

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


class InferenceImageDataMixin(CVDataModule, metaclass=ConfigMeta):
    dataset_base: Type[InferenceImagePathsDataset]

    def __init__(
        self,
        inp: Any,
        *,
        batch_size: int,
        num_workers: int = 0,
        prefetch_device: Optional[Union[int, str]] = None,
        transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
        transform_config: Optional[Dict[str, Any]] = None,
    ):
        self.inp = inp
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
        self.dataset = self.dataset_base(self.inp, self.transform)

    def initialize(self) -> Tuple[CVLoader, Optional[CVLoader]]:
        args = self.batch_size, self.num_workers, self.prefetch_device
        loader = self.dataset.make_loader(*args)
        return loader, None


class InferenceImagePathsData(InferenceImageDataMixin):
    dataset_base = InferenceImagePathsDataset


class InferenceImageFolderData(InferenceImageDataMixin):
    dataset_base = InferenceImageFolderDataset


class PrepareResults(NamedTuple):
    data: ImageFolderData
    tgt_folder: str


class _IPreparation:
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


class DefaultPreparation(_IPreparation):
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
    preparation: _IPreparation = DefaultPreparation(),
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

        print_info(f"saving {dtype} dataset")
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
    preparation: _IPreparation = DefaultPreparation(),
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
    use_auto_label_weights: bool = False,
    label_weights: Optional[Dict[str, float]] = None,
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
    if use_auto_label_weights and label_weights is not None:
        print_warning(
            "`label_weights` is already provided, "
            "so `use_auto_label_weights` will not take effect"
        )
        use_auto_label_weights = False
    label_mapping = None
    if use_auto_label_weights or label_weights is not None:
        label_mapping = {}
        for split in ["train", "valid"]:
            split_folder = os.path.join(tgt_folder, split)
            label_path = os.path.join(split_folder, f"{LABEL_KEY}.json")
            with open(label_path, "r") as f:
                label_mapping.update(json.load(f))
    if use_auto_label_weights:
        labels = list(label_mapping.values())
        unique, counts = np.unique(labels, return_counts=True)
        label_weights = {key: 1.0 / value for key, value in zip(unique, counts)}
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
        label_mapping=label_mapping,
        label_weights=label_weights,
        lmdb_config=lmdb_config,
    )
    return PrepareResults(data, tgt_folder)


__all__ = [
    "LMDBItem",
    "CVDataset",
    "ImageFolderDataset",
    "InferenceImagePathsDataset",
    "InferenceImageFolderDataset",
    "CVLoader",
    "Transforms",
    "Function",
    "Compose",
    "default_lmdb_path",
    "CVDataModule",
    "ImageFolderData",
    "InferenceImagePathsData",
    "InferenceImageFolderData",
    "DefaultPreparation",
    "prepare_image_folder",
    "prepare_image_folder_data",
]
