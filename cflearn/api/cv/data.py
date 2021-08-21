import os
import cv2
import dill
import json
import lmdb
import shutil

import numpy as np
import albumentations as A

from abc import abstractmethod
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
from skimage import io
from functools import partial
from cftool.dist import Parallel
from cftool.misc import is_numeric
from cftool.misc import shallow_copy_dict
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from ...types import tensor_dict_type
from ...types import sample_weights_type
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY
from ...constants import ERROR_PREFIX
from ...constants import WARNING_PREFIX
from ...constants import ORIGINAL_LABEL_KEY
from ...misc.toolkit import to_torch
from ...misc.toolkit import WithRegister
from ...misc.internal_ import DLData
from ...misc.internal_ import DLLoader
from ...misc.internal_ import DataLoader
from ...misc.internal_ import DLDataModule


cf_transforms: Dict[str, Type["Transforms"]] = {}


class Transforms(WithRegister):
    d: Dict[str, Type["Transforms"]] = cf_transforms

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


@Transforms.register("to_tensor")
class ToTensor(Transforms):
    fn = transforms.ToTensor()

    @property
    def need_batch_process(self) -> bool:
        return False


@Transforms.register("random_resized_crop")
class RandomResizedCrop(Transforms):
    def __init__(self, *, size: int = 224):
        super().__init__()
        self.fn = transforms.RandomResizedCrop(size)

    @property
    def need_batch_process(self) -> bool:
        return False


@Transforms.register("-1~1")
class N1To1(Transforms):
    fn = transforms.Lambda(lambda t: t * 2.0 - 1.0)

    @property
    def need_batch_process(self) -> bool:
        return False


@Transforms.register("for_generation")
class ForGeneration(Compose):
    def __init__(self):  # type: ignore
        super().__init__([ToTensor(), N1To1()])


@Transforms.register("for_imagenet")
class ForImagenet(Compose):
    def __init__(self):  # type: ignore
        super().__init__([ToArray(), Resize(224), Normalize(), ToTensor()])


class ATransforms(Transforms):
    input_alias = "image"

    def __init__(self, *, label_alias: Optional[str] = None):
        super().__init__()
        self.label_alias = label_alias

    def __call__(self, inp: Any, **kwargs: Any) -> Any:  # type: ignore
        if not self.need_batch_process:
            kwargs[self.input_alias] = inp
            return self.fn(**kwargs)[self.input_alias]
        inp_keys_mapping = {
            self.input_alias
            if k == INPUT_KEY
            else self.label_alias
            if k == LABEL_KEY
            else k: k
            for k in inp
        }
        inp = {k: inp[v] for k, v in inp_keys_mapping.items()}
        return {inp_keys_mapping[k]: v for k, v in self.fn(**inp).items()}

    @property
    def need_batch_process(self) -> bool:
        return self.label_alias is not None


@Transforms.register("to_array")
class ToArray(ATransforms):
    def __init__(self, *, label_alias: Optional[str] = None):
        super().__init__(label_alias=label_alias)
        self.fn = lambda **inp: {k: np.array(v) for k, v in inp.items()}


@Transforms.register("resize")
class Resize(ATransforms):
    def __init__(
        self,
        size: Union[int, tuple] = 224,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        if isinstance(size, int):
            size = size, size
        self.fn = A.Resize(*size)


@Transforms.register("random_crop")
class RandomCrop(ATransforms):
    def __init__(self, size: Union[int, tuple], *, label_alias: Optional[str] = None):
        super().__init__(label_alias=label_alias)
        if isinstance(size, int):
            size = size, size
        self.fn = A.RandomCrop(*size)


@Transforms.register("shift_scale_rotate")
class ShiftScaleRotate(ATransforms):
    def __init__(
        self,
        p: float = 0.5,
        border_mode: int = cv2.BORDER_REFLECT_101,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.ShiftScaleRotate(border_mode=border_mode, p=p)


@Transforms.register("hflip")
class HFlip(ATransforms):
    def __init__(self, p: float = 0.5, *, label_alias: Optional[str] = None):
        super().__init__(label_alias=label_alias)
        self.fn = A.HorizontalFlip(p=p)


@Transforms.register("vflip")
class VFlip(ATransforms):
    def __init__(self, p: float = 0.5, *, label_alias: Optional[str] = None):
        super().__init__(label_alias=label_alias)
        self.fn = A.VerticalFlip(p=p)


@Transforms.register("normalize")
class Normalize(ATransforms):
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        max_pixel_value: float = 1.0,
        p: float = 1.0,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.Normalize(mean, std, max_pixel_value, p=p)


@Transforms.register("rgb_shift")
class RGBShift(ATransforms):
    def __init__(
        self,
        r_shift_limit: float = 0.08,
        g_shift_limit: float = 0.08,
        b_shift_limit: float = 0.08,
        p: float = 0.5,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.RGBShift(r_shift_limit, g_shift_limit, b_shift_limit, p=p)


@Transforms.register("solarize")
class Solarize(ATransforms):
    def __init__(
        self,
        threshold: float = 0.5,
        p: float = 0.5,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.Solarize(threshold, p=p)


@Transforms.register("gaussian_blur")
class GaussianBlur(ATransforms):
    def __init__(
        self,
        blur_limit: Tuple[int, int] = (3, 7),
        sigma_limit: int = 0,
        p: float = 0.5,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.GaussianBlur(blur_limit, sigma_limit, p=p)


@Transforms.register("hue_saturation")
class HueSaturationValue(ATransforms):
    def __init__(
        self,
        hue_shift_limit: float = 0.08,
        sat_shift_limit: float = 0.12,
        val_shift_limit: float = 0.08,
        p: float = 0.5,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.HueSaturationValue(
            hue_shift_limit,
            sat_shift_limit,
            val_shift_limit,
            p,
        )


@Transforms.register("brightness_contrast")
class RandomBrightnessContrast(ATransforms):
    def __init__(
        self,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        brightness_by_max: bool = True,
        p: float = 0.5,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.RandomBrightnessContrast(
            brightness_limit,
            contrast_limit,
            brightness_by_max,
            p,
        )


@Transforms.register("a_to_tensor")
class AToTensor(ATransforms):
    def __init__(
        self,
        transpose_mask: bool = True,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = ToTensorV2(transpose_mask)


@Transforms.register("a_bundle")
class ABundle(Compose):
    def __init__(
        self,
        *,
        resize_size: int = 320,
        crop_size: int = 288,
        p: float = 0.5,
        label_alias: Optional[str] = None,
    ):
        super().__init__(
            [
                Resize(resize_size, label_alias=label_alias),
                RandomCrop(crop_size, label_alias=label_alias),
                HFlip(p, label_alias=label_alias),
                VFlip(p, label_alias=label_alias),
                ShiftScaleRotate(p, cv2.BORDER_CONSTANT, label_alias=label_alias),
                RGBShift(p=p, label_alias=label_alias),
                Solarize(p=p, label_alias=label_alias),
                GaussianBlur(p=p, label_alias=label_alias),
                HueSaturationValue(p=p, label_alias=label_alias),
                RandomBrightnessContrast(p=p, label_alias=label_alias),
                Normalize(label_alias=label_alias),
                AToTensor(label_alias=label_alias),
            ]
        )


@Transforms.register("a_bundle_test")
class ABundleTest(Compose):
    def __init__(self, *, resize_size: int = 320, label_alias: Optional[str] = None):
        super().__init__(
            [
                Resize(resize_size, label_alias=label_alias),
                Normalize(label_alias=label_alias),
                AToTensor(label_alias=label_alias),
            ]
        )


def batch_callback(
    label_callback: Optional[Callable[[Tuple[Tensor, Tensor]], Tensor]],
    batch: Tuple[Tensor, Tensor],
) -> tensor_dict_type:
    img, labels = batch
    if label_callback is None:
        actual_labels = labels.view(-1, 1)
    else:
        actual_labels = label_callback(batch)
    return {
        INPUT_KEY: img,
        LABEL_KEY: actual_labels,
        ORIGINAL_LABEL_KEY: labels,
    }


class MNISTData(DLDataModule):
    def __init__(
        self,
        *,
        root: str = "data",
        shuffle: bool = True,
        batch_size: int = 64,
        transform: Optional[Union[str, List[str], "Transforms", Callable]],
        transform_config: Optional[Dict[str, Any]] = None,
        label_callback: Optional[Callable[[Tuple[Tensor, Tensor]], Tensor]] = None,
    ):
        self.root = root
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.transform = Transforms.convert(transform, transform_config)
        self.label_callback = label_callback

    # TODO : support sample weights
    def prepare(self, sample_weights: sample_weights_type) -> None:
        self.train_data = DLData(
            MNIST(
                self.root,
                transform=self.transform,
                download=True,
            )
        )
        self.valid_data = DLData(
            MNIST(
                self.root,
                train=False,
                transform=self.transform,
                download=True,
            )
        )

    def initialize(self) -> Tuple[DLLoader, Optional[DLLoader]]:
        train_loader = DLLoader(
            DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            ),
            partial(batch_callback, self.label_callback),
        )
        valid_loader = DLLoader(
            DataLoader(
                self.valid_data,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            ),
            partial(batch_callback, self.label_callback),
        )
        return train_loader, valid_loader


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


def get_tensor_loader(
    x: Tensor,
    y: Optional[Tensor],
    others: Optional[tensor_dict_type] = None,
    *,
    shuffle: bool = True,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DLLoader:
    data = DLData(TensorDataset(x, y, others))
    return DLLoader(DataLoader(data, batch_size, shuffle, num_workers=num_workers))  # type: ignore


def get_tensor_loaders(
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
) -> Tuple[DLLoader, Optional[DLLoader]]:
    base_kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    train_loader = get_tensor_loader(x_train, y_train, train_others, **base_kwargs)  # type: ignore
    if x_valid is None:
        return train_loader, None
    valid_loader = get_tensor_loader(x_valid, y_valid, valid_others, **base_kwargs)  # type: ignore
    return train_loader, valid_loader


class LMDBItem(NamedTuple):
    image: np.ndarray
    labels: Dict[str, Any]


def _default_lmdb_path(folder: str, split: str) -> str:
    return f"{folder}.{split}"


def prepare_image_folder(
    src_folder: str,
    tgt_folder: str,
    *,
    to_index: bool,
    label_fn: Callable[[List[str]], Any],
    filter_fn: Optional[Callable[[List[str]], bool]] = None,
    force_rerun: bool = False,
    src_prepare_fn: Optional[Callable[[str], None]] = None,
    extra_label_names: Optional[List[str]] = None,
    extra_label_fns: Optional[List[Callable[[List[str]], Any]]] = None,
    extensions: Optional[Set[str]] = None,
    make_labels_in_parallel: bool = False,
    num_jobs: int = 8,
    train_all_data: bool = False,
    valid_split: Union[int, float] = 0.1,
    max_num_valid: int = 10000,
    copy_fn: Optional[Callable[[str, str], None]] = None,
    get_img_path_fn: Optional[Callable[[int, str, str], str]] = None,
    lmdb_configs: Optional[Dict[str, Any]] = None,
    use_tqdm: bool = True,
) -> None:
    if not force_rerun and all(
        os.path.isfile(os.path.join(tgt_folder, split, "labels.json"))
        for split in ["train", "valid"]
    ):
        return None

    if src_prepare_fn is not None:
        src_prepare_fn(src_folder)
    os.makedirs(tgt_folder, exist_ok=True)

    walked = list(os.walk(src_folder))
    print("> collecting hierarchies")
    all_img_paths = []
    hierarchy_list = []
    if extensions is None:
        extensions = {".jpg", ".png"}
    for folder, _, files in tqdm(walked, desc="folders", position=0):
        for file in tqdm(files, desc="files", position=1, leave=False):
            if not any(file.endswith(ext) for ext in extensions):
                continue
            hierarchy = folder.split(os.path.sep) + [file]
            if filter_fn is not None and not filter_fn(hierarchy):
                continue
            hierarchy_list.append(hierarchy)
            all_img_paths.append(os.path.join(folder, file))

    def get_labels(fn: Callable[[List[str]], Any]) -> List[Any]:
        def task(h: List[str]) -> Any:
            try:
                return fn(h)
            except Exception as err:
                err_path = "/".join(h)
                msg = f"error occurred ({err}) when getting label of {err_path}"
                print(f"{ERROR_PREFIX}{msg}")
                return None

        if not make_labels_in_parallel:
            return [task(h) for h in tqdm(hierarchy_list)]
        parallel = Parallel(num_jobs, use_tqdm=use_tqdm)
        groups = parallel.grouped(task, hierarchy_list).ordered_results
        return sum(groups, [])

    print("> making labels")
    labels = get_labels(label_fn)
    excluded_indices = {i for i, label in enumerate(labels) if label is None}
    extra_labels_dict: Optional[Dict[str, List[str]]] = None
    if extra_label_names is not None:
        extra_labels_dict = {}
        if extra_label_fns is None:
            raise ValueError(
                "`extra_label_fns` should be provided "
                "when `extra_label_names` is provided"
            )
        print("> making extra labels")
        for el_name, el_fn in zip(extra_label_names, extra_label_fns):
            extra_labels_dict[el_name] = get_labels(el_fn)
        for extra_labels in extra_labels_dict.values():
            for i, extra_label in enumerate(extra_labels):
                if extra_label is None:
                    excluded_indices.add(i)

    if not to_index:
        labels_dict = {"": labels}
    else:
        label2idx = {label: i for i, label in enumerate(sorted(set(labels)))}
        with open(os.path.join(tgt_folder, "label2idx.json"), "w") as f:
            json.dump(label2idx, f)
        with open(os.path.join(tgt_folder, "idx2label.json"), "w") as f:
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

    shuffled_indices = np.random.permutation(num_sample)
    if train_all_data:
        train_indices = shuffled_indices
    else:
        train_indices = shuffled_indices[:-valid_split]
    valid_indices = shuffled_indices[-valid_split:]

    if copy_fn is None:
        copy_fn = lambda src, tgt: shutil.copy(src, tgt)
    if get_img_path_fn is None:

        def get_img_path_fn(i: int, split_folder: str, src_img_path: str) -> str:
            ext = os.path.splitext(src_img_path)[1]
            return os.path.join(split_folder, f"{i}{ext}")

    def save(indices: np.ndarray, d_num_jobs: int, dtype: str) -> None:
        def record(idx: int) -> Optional[Tuple[str, Dict[str, Any]]]:
            assert copy_fn is not None
            assert get_img_path_fn is not None
            split_folder = os.path.join(tgt_folder, dtype)
            os.makedirs(split_folder, exist_ok=True)
            img_path = all_img_paths[idx]
            new_img_path = get_img_path_fn(idx, split_folder, img_path)
            try:
                copy_fn(img_path, new_img_path)
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
        results = sum(parallel.grouped(record, indices).ordered_results, [])
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
        for label_type, type_labels in merged_labels.items():
            delim = "_" if label_type else ""
            label_file = f"{label_type}{delim}labels.json"
            with open(os.path.join(tgt_folder, dtype, label_file), "w") as f_:
                json.dump(type_labels, f_)
        # lmdb
        if lmdb_configs is None:
            return None
        local_lmdb_configs = shallow_copy_dict(lmdb_configs)
        local_lmdb_configs.setdefault("path", _default_lmdb_path(tgt_folder, dtype))
        local_lmdb_configs.setdefault("map_size", 1099511627776 * 2)
        db = lmdb.open(**local_lmdb_configs)
        context = db.begin(write=True)
        d_num_samples = len(indices)
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
                dill.dumps(LMDBItem(io.imread(path), i_new_labels)),
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


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        folder: str,
        split: str,
        transform: Optional[Union[str, List[str], "Transforms", Callable]],
        transform_config: Optional[Dict[str, Any]] = None,
        lmdb_configs: Optional[Dict[str, Any]] = None,
    ):
        if lmdb_configs is not None:
            self.lmdb_configs = shallow_copy_dict(lmdb_configs)
            self.lmdb_configs.setdefault("path", _default_lmdb_path(folder, split))
            self.lmdb_configs.setdefault("lock", False)
            self.lmdb_configs.setdefault("meminit", False)
            self.lmdb_configs.setdefault("readonly", True)
            self.lmdb_configs.setdefault("readahead", False)
            self.lmdb_configs.setdefault("max_readers", 32)
            self.lmdb = lmdb.open(**self.lmdb_configs)
            self.context = self.lmdb.begin(buffers=True, write=False)
            with self.lmdb.begin(write=False) as context:
                self.length = int(context.get("length".encode("ascii")).decode("ascii"))
        else:
            self.lmdb = self.context = None
            self.folder = os.path.abspath(os.path.join(folder, split))
            with open(os.path.join(self.folder, "labels.json"), "r") as f:
                self.labels = json.load(f)
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
            self.length = len(self.img_paths)
        self.transform = Transforms.convert(transform, transform_config)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # TODO : Change this to self.context once found out why CI breaks down
        if getattr(self, "context", None) is not None:
            key = str(index).encode("ascii")
            item: LMDBItem = dill.loads(self.context.get(key))
            img = Image.fromarray(item.image)
            label = item.labels
            if len(label) == 1:
                label = list(label.values())[0]
        else:
            file = self.img_paths[index]
            img = Image.open(file)
            label = self.labels[file]
            if is_numeric(label):
                label = np.array([label])
            elif isinstance(label, str) and label.endswith(".npy"):
                label = np.load(label)
        if self.transform is None:
            img = to_torch(np.array(img).astype(np.float32))
            if isinstance(label, np.ndarray):
                label = to_torch(label)
            return {INPUT_KEY: img, LABEL_KEY: label}
        if self.transform.need_batch_process:
            img_arr = np.array(img).astype(np.float32) / 255.0
            return self.transform({INPUT_KEY: img_arr, LABEL_KEY: label})
        if isinstance(label, np.ndarray):
            label = to_torch(label)
        return {INPUT_KEY: self.transform(img), LABEL_KEY: label}

    def __len__(self) -> int:
        return self.length


def get_image_folder_loaders(
    folder: str,
    *,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    transform: Optional[Union[str, Transforms]] = None,
    test_shuffle: Optional[bool] = None,
    test_transform: Optional[Union[str, Transforms]] = None,
    lmdb_configs: Optional[Dict[str, Any]] = None,
) -> Tuple[DLLoader, DLLoader]:
    train_data = DLData(
        ImageFolderDataset(
            folder,
            "train",
            transform,
            lmdb_configs=lmdb_configs,
        )
    )
    valid_data = DLData(
        ImageFolderDataset(
            folder,
            "valid",
            test_transform or transform,
            lmdb_configs=lmdb_configs,
        )
    )
    base_kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    train_loader = DLLoader(DataLoader(train_data, **base_kwargs))  # type: ignore
    if test_shuffle is None:
        test_shuffle = shuffle
    base_kwargs["shuffle"] = test_shuffle
    valid_loader = DLLoader(DataLoader(valid_data, **base_kwargs))  # type: ignore
    return train_loader, valid_loader
