import os
import dill
import json
import random
import shutil

import numpy as np

from abc import abstractmethod
from PIL import Image
from enum import Enum
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
from cftool.cv import to_rgb
from cftool.dist import Parallel
from cftool.misc import walk
from cftool.misc import hash_dict
from cftool.misc import is_numeric
from cftool.misc import print_info
from cftool.misc import print_error
from cftool.misc import print_warning
from cftool.misc import get_arguments
from cftool.misc import shallow_copy_dict
from cftool.misc import ISerializable

from ....schema import IDataset
from ....schema import IDataBlock
from ....schema import DataBundle
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY

try:
    import lmdb
except:
    lmdb = None


READY_FILE = "READY"
preparations: Dict[str, Type["IPreparation"]] = {}
default_image_extensions = {".jpg", ".png", ".jpeg"}


class IPreparation(ISerializable):
    d = preparations

    # abstract

    @abstractmethod
    def filter(self, hierarchy: List[str]) -> bool:
        """return whether the `hierarchy` should be abandoned"""

    @abstractmethod
    def get_label(self, hierarchy: List[str]) -> Any:
        """
        return the label of the `hierarchy`
        * notice that the returned label should be hashable
        """

    @abstractmethod
    def copy(self, src_path: str, tgt_path: str) -> None:
        """
        copy an image from `src_path` to `tgt_path`.
        * If any error occurred in this method, we will handle it and filter out the
        corresponding sample, so feel free to do preprocessing or do sanity checks
        here and raise errors!
        """

    # optional callbacks

    @property
    def extra_labels(self) -> Optional[List[str]]:
        """
        overwrite this property if extra labels are needed.
        * each of the returned element will be passed to the `get_extra_label` method
        below as the `label_name` argument.
        """
        return None

    def prepare_src_folder(self, src_folder: str) -> None:
        """
        prepare the source folder before other processes happen.
        * common use case is to 'augment' the source folder beforehand.
        """

    def get_extra_label(self, label_name: str, hierarchy: List[str]) -> Any:
        """
        return extra label.
        * will not be called if the above `extra_labels` property returns `None`.
        """

    def to_info(self) -> Dict[str, Any]:
        return {}

    def from_info(self, info: Dict[str, Any]) -> None:
        """
        `IPreparation` will not be created from scratch, and will always be
        instantiated with the `from_pack` method. So if we need to assign
        some attributes, we need to do them here.
        * See `ResizedPreparation` below as an example.
        """

    # api

    def is_ready(
        self,
        tgt_folder: str,
        valid_split: Union[int, float],
        folder_hash: str,
    ) -> bool:
        candidates = [DatasetSplit.TRAIN]
        if valid_split > 0:
            candidates.append(DatasetSplit.VALID)
        for split in candidates:
            split_folder = os.path.join(tgt_folder, split)
            extra_keys = [f"{key}_{LABEL_KEY}" for key in self.extra_labels or []]
            for key in [LABEL_KEY] + extra_keys:
                path = os.path.join(split_folder, f"{key}.json")
                if not os.path.isfile(path):
                    return False
            ready_path = os.path.join(split_folder, READY_FILE)
            if not os.path.isfile(ready_path):
                return False
            with open(ready_path, "r") as f:
                if f.read().strip() != folder_hash:
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


@IPreparation.register("default")
class DefaultPreparation(IPreparation):
    """
    A default preparation.
    * All images are treated as valid.
    * No labels will be calculated.
    """

    def filter(self, hierarchy: List[str]) -> bool:
        return True

    def get_label(self, hierarchy: List[str]) -> Any:
        return 0

    def copy(self, src_path: str, tgt_path: str) -> None:
        shutil.copyfile(src_path, tgt_path)


@IPreparation.register("resized")
class ResizedPreparation(DefaultPreparation):
    """
    A Preparation with images resized.
    * Useful when we need a static-sized image dataset.
    * The attribute, `img_size`, should be initialized when `from_info` is called.
    """

    img_size: int
    keep_aspect_ratio: bool

    def copy(self, src_path: str, tgt_path: str) -> None:
        img = to_rgb(Image.open(src_path))
        if not self.keep_aspect_ratio:
            img.resize((self.img_size, self.img_size), Image.LANCZOS).save(tgt_path)
            return None
        w, h = img.size
        wh_ratio = w / h
        if wh_ratio < 1:
            new_w = self.img_size
            new_h = round(self.img_size / wh_ratio)
        else:
            new_w = round(self.img_size * wh_ratio)
            new_h = self.img_size
        img.resize((new_w, new_h), Image.LANCZOS).save(tgt_path)

    def to_info(self) -> Dict[str, Any]:
        return dict(img_size=self.img_size, keep_aspect_ratio=self.keep_aspect_ratio)

    def from_info(self, info: Dict[str, Any]) -> None:
        img_size = info.get("img_size")
        if img_size is None:
            raise ValueError("`img_size` should be provided for `ResizedPreparation`")
        self.img_size = img_size
        self.keep_aspect_ratio = info.setdefault("keep_aspect_ratio", True)


def default_lmdb_path(folder: str, split: str) -> str:
    return f"{folder}.{split}"


class LMDBItem(NamedTuple):
    image: np.ndarray
    labels: Dict[str, Any]


def prepare_image_folder(
    *,
    src_folder: str,
    tgt_folder: str,
    to_index: bool,
    prefix: Optional[str],
    preparation_pack: Optional[Dict[str, Any]],
    force_rerun: bool,
    extensions: Optional[Set[str]],
    make_labels_in_parallel: bool,
    save_data_in_parallel: bool,
    num_jobs: int,
    train_all_data: bool,
    valid_split: Union[int, float],
    max_num_valid: int,
    lmdb_config: Optional[Dict[str, Any]],
    use_tqdm: bool,
    strict: bool,
) -> Tuple[str, Optional[List[str]]]:
    f"""
    Efficiently convert an arbitrary image folder (`src_folder`) to
    an image folder (`tgt_folder`) which can be parsed by `carefree-learn`.
    * It offers various customization options such as parallel processing,
    validation split, and LMDB integration.

    Parameters
    ----------
    src_folder : str
        The path to the source folder containing the images to be processed.
    tgt_folder : str
        The path to the target folder where the processed images and related metadata will be saved.
    to_index : bool
        Whether to convert the labels to indices or not.
    prefix : Optional[str], default: None
        An optional prefix for the source and target folder paths.
    preparation_pack : Optional[Dict[str, Any]], default: None
        An optional JSON configuration for customizing the image preparation process.
        * should be of format: dict(type="...", info=dict(...))
    force_rerun : bool
        If True, the function will rerun even if the target folder already exists and is ready.
    extensions : Optional[Set[str]], default: None
        A set of allowed image file extensions. Defaults to {default_image_extensions} if not provided.
    make_labels_in_parallel : bool
        Whether to create labels in parallel or not.
    save_data_in_parallel : bool
        Whether to save data in parallel or not.
    num_jobs : int
        The number of jobs to use for parallel processing.
    train_all_data : bool
        If True, all data will be used for training, and no validation set will be created.
    valid_split : Union[int, float]
        The number or proportion of samples to be used for validation. If a float is provided,
        it represents the proportion of the dataset to include in the validation set.
    max_num_valid : int
        The maximum number of samples to be used for validation.
    lmdb_config : Optional[Dict[str, Any]], default: None
        An optional dictionary containing LMDB configuration. If not provided, LMDB will not be used.
    use_tqdm : bool
        If True, display progress bars using `tqdm`.
    strict : bool
        If True, error will be raised if any invalid sample occurred.

    Returns
    -------
    str
        The path to the target folder containing the processed images and related metadata.
    """

    if prefix is not None:
        src_folder = os.path.join(prefix, src_folder)
        tgt_folder = os.path.join(prefix, tgt_folder)
    if preparation_pack is None:
        preparation_pack = dict(type="default", info={})
    preparation: IPreparation = IPreparation.from_pack(preparation_pack)
    preparation.prepare_src_folder(src_folder)

    if train_all_data:
        valid_split = 0
    folder_hash = hash_dict(
        dict(
            src_folder=src_folder,
            to_index=to_index,
            preparation_pack=preparation_pack,
            extensions=extensions,
            train_all_data=train_all_data,
            valid_split=valid_split,
            max_num_valid=max_num_valid,
            lmdb_config=lmdb_config,
            strict=strict,
        )
    )
    if not force_rerun and preparation.is_ready(tgt_folder, valid_split, folder_hash):
        return tgt_folder, preparation.extra_labels

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
        extensions = default_image_extensions
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

    # exclude samples
    if excluded_indices:
        if not strict:
            print_warning(f"{len(excluded_indices)} samples will be excluded")
        else:
            raise ValueError(
                "\n".join(
                    [
                        "following samples are invalid:",
                        *[f"* {all_img_paths[i]}" for i in excluded_indices],
                        "please check the log history for more details",
                    ]
                )
            )
    for i in sorted(excluded_indices)[::-1]:
        labels.pop(i)
        if extra_labels_dict is not None:
            for sub_labels in extra_labels_dict.values():
                sub_labels.pop(i)
        all_img_paths.pop(i)

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

    # prepare core
    def save(indices: np.ndarray, d_num_jobs: int, dtype: str) -> None:
        def record(idx: int) -> Optional[Tuple[str, Dict[str, Any]]]:
            split_folder = os.path.join(tgt_folder, dtype)
            os.makedirs(split_folder, exist_ok=True)
            img_path = all_img_paths[idx]
            ext = os.path.splitext(img_path)[1]
            new_img_path = os.path.join(split_folder, f"{idx}{ext}")
            try:
                preparation.copy(img_path, new_img_path)
                key = os.path.abspath(new_img_path)
                idx_labels: Dict[str, Any] = {}
                for label_t, t_labels in labels_dict.items():
                    idx_labels[label_t] = {key: t_labels[idx]}
                return key, idx_labels
            except Exception as err:
                print_error(f"error occurred with {img_path} : {err}")
                return None

        print_info(f"saving {dtype} dataset")
        results: List[Tuple[str, Dict[str, Any]]]
        indices = indices.copy()
        np.random.shuffle(indices)
        if not save_data_in_parallel:
            results = [record(i) for i in indices]  # type: ignore
        else:
            parallel = Parallel(d_num_jobs, use_tqdm=use_tqdm)
            results = sum(parallel.grouped(record, indices).ordered_results, [])
        d_valid_indices = [i for i, r in enumerate(results) if r is not None]
        results = [results[i] for i in d_valid_indices]
        valid_paths = [all_img_paths[idx] for idx in indices[d_valid_indices]]
        new_paths, all_labels_list = zip(*results)
        merged_labels = shallow_copy_dict(all_labels_list[0])
        for sub_labels_ in all_labels_list[1:]:
            for k, v in shallow_copy_dict(sub_labels_).items():
                merged_labels[k].update(v)
        print_info(
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
                msg = "`lmdb` is not installed, so `lmdb_config` will be ignored"
                print_warning(msg)
        else:
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
        # dump READY
        with open_dtype_file(READY_FILE) as f_:
            f_.write(folder_hash)

    num_sample = len(all_img_paths)
    if isinstance(valid_split, float):
        if valid_split < 0.0 or valid_split >= 1.0:
            raise ValueError("`valid_split` should be within [0, 1)")
        valid_split = max(1, min(max_num_valid, int(round(num_sample * valid_split))))
    assert isinstance(valid_split, int)

    if valid_split <= 0:
        save(np.arange(num_sample), max(1, num_jobs), DatasetSplit.TRAIN)
        return tgt_folder, extra_labels

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
    train_indices = np.array(merged_train_indices)
    valid_indices = np.array(merged_valid_indices)

    save(train_indices, max(1, num_jobs), DatasetSplit.TRAIN)
    save(valid_indices, max(1, num_jobs // 2), DatasetSplit.VALID)
    return tgt_folder, extra_labels


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALID = "valid"


class ImagePaths(NamedTuple):
    length: int
    img_paths: List[str]
    # label_name -> path (from `img_paths`) -> label (of the corresponding path)
    labels: Optional[Dict[str, Dict[str, Any]]]
    lmdb_context: Optional[Any]

    def to_dataset(self) -> "ImageFolderDataset":
        return ImageFolderDataset(self)


def get_image_paths(
    split: DatasetSplit,
    base_folder: str,
    extra_label_names: Optional[List[str]],
    lmdb_config: Optional[Dict[str, Any]],
) -> Optional[ImagePaths]:
    folder = os.path.abspath(os.path.join(base_folder, split))
    if not os.path.isdir(folder):
        return None
    support_extensions = default_image_extensions
    support_extensions.add(".npy")
    img_paths = list(
        map(
            lambda file: os.path.join(folder, file),
            filter(
                lambda file: os.path.splitext(file)[1] in support_extensions,
                os.listdir(folder),
            ),
        )
    )
    if lmdb_config is None or lmdb is None:
        if lmdb_config is not None:
            print_warning("`lmdb` is not installed, so `lmdb_config` will be ignored")
        lmdb_context = None
        with open(os.path.join(folder, f"{LABEL_KEY}.json"), "r") as f:
            labels = {LABEL_KEY: json.load(f)}
        if extra_label_names is not None:
            for name in extra_label_names:
                el_file = f"{name}_{LABEL_KEY}.json"
                with open(os.path.join(folder, el_file), "r") as f:
                    labels[name] = json.load(f)
        length = len(img_paths)
    else:
        new_lmdb_config = shallow_copy_dict(lmdb_config)
        new_lmdb_config.setdefault("path", default_lmdb_path(base_folder, split))
        new_lmdb_config.setdefault("lock", False)
        new_lmdb_config.setdefault("meminit", False)
        new_lmdb_config.setdefault("readonly", True)
        new_lmdb_config.setdefault("readahead", False)
        new_lmdb_config.setdefault("max_readers", 32)
        lmdb_ = lmdb.open(**new_lmdb_config)
        lmdb_context = lmdb_.begin(buffers=True, write=False)
        with lmdb_.begin(write=False) as context:
            length = int(context.get("length".encode("ascii")).decode("ascii"))
    return ImagePaths(length, img_paths, labels, lmdb_context)


class ImageFolderDataset(IDataset):
    def __init__(self, paths: ImagePaths) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return self.paths.length

    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> Dict[str, Any]:
        if not isinstance(item, int):
            raise ValueError("`ImageFolderDataset` only supports single index")
        if self.paths.lmdb_context is not None:
            key = str(item).encode("ascii")
            lmdb_item: LMDBItem = dill.loads(self.paths.lmdb_context.get(key))
            new_item = lmdb_item.labels
            for k, v in new_item.items():
                if is_numeric(v):
                    new_item[k] = np.array([v])
            new_item[INPUT_KEY] = lmdb_item.image
            return new_item
        img_path = self.paths.img_paths[item]
        image = np.array(Image.open(img_path))
        if self.paths.labels is None:
            return {INPUT_KEY: image}
        new_item = {k: v[img_path] for k, v in self.paths.labels.items()}
        for k, v in new_item.items():
            if is_numeric(v):
                v = np.array([v])
            elif isinstance(v, str) and v.endswith(".npy"):
                v = np.load(v)
            new_item[k] = v
        new_item[INPUT_KEY] = image
        return new_item


@IDataBlock.register("image_folder")
class ImageFolderBlock(IDataBlock):
    """
    This block will utilize `prepare_image_folder` method to create an image folder
    which can be parsed by `carefree-learn`.

    It will then transfer the input `DataBundle`, which should only contain `x_train` indicating
    the original image folder, to a new `DataBundle` containing the `ImageFolderDataset`(s).

    TODO: Support parsing `x_valid` and use it directly as validation `ImageFolderDataset`.
    """

    tgt_folder: Optional[str]
    to_index: bool
    prefix: Optional[str]
    preparation_pack: Optional[Dict[str, Any]]
    force_rerun: bool
    extensions: Optional[Set[str]]
    make_labels_in_parallel: bool
    save_data_in_parallel: bool
    num_jobs: int
    train_all_data: bool
    valid_split: Union[int, float]
    max_num_valid: int
    lmdb_config: Optional[Dict[str, Any]]
    use_tqdm: bool
    strict: bool

    def __init__(
        self,
        *,
        tgt_folder: Optional[str] = None,
        to_index: bool = False,
        prefix: Optional[str] = None,
        preparation_pack: Optional[Dict[str, Any]] = None,
        force_rerun: bool = False,
        extensions: Optional[Set[str]] = None,
        make_labels_in_parallel: bool = False,
        save_data_in_parallel: bool = True,
        num_jobs: int = 8,
        train_all_data: bool = False,
        valid_split: Union[int, float] = 0.1,
        max_num_valid: int = 10000,
        lmdb_config: Optional[Dict[str, Any]] = None,
        use_tqdm: bool = True,
        strict: bool = False,
    ) -> None:
        super().__init__(**get_arguments())

    # inherit

    @property
    def fields(self) -> List[str]:
        return [
            "tgt_folder",
            "to_index",
            "prefix",
            "preparation_pack",
            "force_rerun",
            "extensions",
            "make_labels_in_parallel",
            "save_data_in_parallel",
            "num_jobs",
            "train_all_data",
            "valid_split",
            "max_num_valid",
            "lmdb_config",
            "use_tqdm",
            "strict",
        ]

    def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
        train_paths, valid_paths = self._parse(bundle.x_train, for_inference)  # type: ignore
        bundle.x_train = train_paths.to_dataset()
        if valid_paths is not None:
            bundle.x_valid = valid_paths.to_dataset()
        return bundle

    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        return self.transform(bundle, False)

    # internal

    def _parse(
        self,
        src_folder: str,
        for_inference: bool,
    ) -> Tuple[ImagePaths, Optional[ImagePaths]]:
        if src_folder is None:
            return None
        info = shallow_copy_dict(self.to_info())
        # get tgt folder
        if self.tgt_folder is None:
            tgt_folder = f"{src_folder}_{'inference' if for_inference else 'training'}"
        else:
            if not for_inference:
                tgt_folder = self.tgt_folder
            else:
                tgt_folder = f"{self.tgt_folder}_inference"
        # set properties
        info["src_folder"] = src_folder
        info["tgt_folder"] = tgt_folder
        if for_inference:
            info["train_all_data"] = True
        # prepare
        tgt_folder, extra_labels = prepare_image_folder(**info)
        # get paths
        common_args = tgt_folder, extra_labels, self.lmdb_config
        train_paths = get_image_paths(DatasetSplit.TRAIN, *common_args)
        valid_paths = get_image_paths(DatasetSplit.VALID, *common_args)
        return train_paths, valid_paths  # type: ignore


__all__ = [
    "default_image_extensions",
    "IPreparation",
    "DefaultPreparation",
    "ResizedPreparation",
    "ImageFolderBlock",
]
