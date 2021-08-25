import os
import dill
import json
import lmdb
import shutil

import numpy as np

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
from cftool.misc import is_numeric
from cftool.misc import shallow_copy_dict
from torch.utils.data import Dataset

from .transforms import Transforms
from ....types import tensor_dict_type
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import ERROR_PREFIX
from ....constants import WARNING_PREFIX
from ....constants import ORIGINAL_LABEL_KEY
from ....misc.toolkit import to_torch
from ....misc.internal_ import CVLoader
from ....misc.internal_ import DataLoader


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


class ImageFolderDataset(Dataset):
    lmdb = None
    context = None
    lmdb_configs = None

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
        if self.lmdb_configs is not None:
            if self.lmdb is None:
                self.lmdb = lmdb.open(**self.lmdb_configs)
            if self.context is None:
                self.context = self.lmdb.begin(buffers=True, write=False)
        if self.context is not None:
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


class InferenceImageFolderDataset(Dataset):
    def __init__(self, folder: str, transform: Optional[Callable]):
        self.folder = os.path.abspath(folder)
        support_appendix = {".jpg", ".png"}
        self.img_paths = list(
            map(
                lambda file: os.path.join(self.folder, file),
                filter(
                    lambda file: file[-4:] in support_appendix,
                    os.listdir(self.folder),
                ),
            )
        )
        self.transform = transform

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        img = Image.open(self.img_paths[index])
        if self.transform is None:
            img = to_torch(np.array(img).astype(np.float32))
            return {INPUT_KEY: img}
        return {INPUT_KEY: self.transform(img)}

    def __len__(self) -> int:
        return len(self.img_paths)

    def make_loader(self, batch_size: int, num_workers: int = 0) -> CVLoader:
        return CVLoader(DataLoader(self, batch_size, num_workers=num_workers))
