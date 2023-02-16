import os
import json
import math

import numpy as np

from abc import abstractmethod
from tqdm import tqdm
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Protocol
from typing import NamedTuple
from collections import Counter
from dataclasses import dataclass
from cftool.dist import Parallel
from cftool.misc import grouped
from cftool.misc import hash_code
from cftool.misc import print_info
from cftool.misc import print_warning
from cftool.misc import DataClassBase
from cftool.array import get_unique_indices
from cftool.array import SharedArray
from cftool.array import StrideArray

from .basic import register_ml_data
from .basic import register_ml_data_processor
from .basic import IMLData
from .basic import IMLBatch
from .basic import MLDatasetTag
from .basic import IMLDataProcessor
from .basic import IMLPreProcessedData


TIME_SERIES_NAME = "_internal.ts"


@dataclass
class TimeSeriesConfig(DataClassBase):
    gap: int
    x_window: int
    y_window: int
    id_column: int
    time_columns: List[int]
    # `validation_split` will be included in the validation dataset
    validation_split: Optional[int] = None
    # `validation_end` will NOT be included in the validation dataset
    validation_end: Optional[int] = None
    # `test_split` will be included in the test dataset
    test_split: Optional[int] = None
    # `test_end` will NOT be included in the test dataset
    test_end: Optional[int] = None
    num_test: Optional[int] = None
    enforce_num_test: bool = False
    enforce_test_valid: bool = False
    num_workers: Optional[int] = None
    random_sample_ratio: Optional[float] = None
    for_inference: bool = False
    sanity_check: bool = False
    no_cache: bool = False
    verbose: bool = True
    cache_folder: str = "cache"

    @property
    def span(self) -> int:
        if self.for_inference:
            return self.x_window
        return self.x_window + self.y_window + self.gap

    @property
    def hash(self) -> str:
        d = self.asdict()
        for k in self.hash_excludes:
            d.pop(k)
        return hash_code(str({k: d[k] for k in sorted(d)}))

    @property
    def hash_excludes(self) -> List[str]:
        excludes = [
            "num_workers",
            "random_sample_ratio",
            "sanity_check",
            "no_cache",
            "verbose",
            "cache_folder",
        ]
        if not self.for_inference:
            excludes += [
                "test_split",
                "test_end",
                "num_test",
                "enforce_num_test",
                "enforce_test_valid",
            ]
        return excludes

    @property
    def use_validation(self) -> bool:
        return not self.for_inference and self.validation_split is not None


@dataclass
class TimeSeriesCachePaths(DataClassBase):
    data_path: str
    merged_indices_path: str
    split_indices_path: str
    valid_indices_path: str

    @property
    def folder(self) -> str:
        return os.path.dirname(self.data_path)

    @property
    def all_exists(self) -> bool:
        return all(map(os.path.isfile, self.asdict().values()))


class TimeSeriesDataBundle(NamedTuple):
    config: TimeSeriesConfig
    # all (flattened) data, should already be sorted by time
    # shape: [N, d]
    data: np.ndarray
    # first hierarchy: sorted ids
    # second hierarchy: indices of each id that point to the id's data
    split_indices: List[List[int]]
    # all rolled indices (including those which contain different ids)
    # shape: [N - span + 1, span]
    rolled_indices: np.ndarray
    # indices of the `rolled_indices` which are 'valid'
    # 1. len(valid_indices) == number of valid samples
    # 2. sample procedure:
    #      i = randint(n)
    #   -> idx = valid_indices[i]
    #   -> indices = rolled_indices[idx]
    #   -> sample = data[indices]
    valid_indices: np.ndarray

    def __len__(self) -> int:
        return len(self.valid_indices)

    def fetch_batch(self, indices: Union[int, List[int], np.ndarray]) -> IMLBatch:
        indices_mat = self.rolled_indices[self.valid_indices[indices]]
        if isinstance(indices, int) or np.isscalar(indices):
            data_batch = self.data[indices_mat]
            x_batch = data_batch[: self.config.x_window]
        else:
            shape = [len(indices), self.config.span, -1]
            data_batch = self.data[indices_mat.ravel()].reshape(shape)
            x_batch = data_batch[:, : self.config.x_window]
        if self.config.for_inference:
            y_batch = None
        else:
            y_batch = data_batch[:, self.config.x_window + self.config.gap :]
        return IMLBatch(
            x_batch,
            None if y_batch is None else y_batch,
            others=dict(full=data_batch),
        )

    def to_loader(
        self,
        *,
        batch_size: Optional[int] = None,
        tqdm_desc: Optional[str] = None,
    ) -> Iterable[IMLBatch]:
        n = len(self)
        it: Iterable = range(n)
        if batch_size is not None:
            n = math.ceil(n / batch_size)
            it = iter(list(i) for i in grouped(it, batch_size, keep_tail=True))
        if tqdm_desc is not None:
            it = tqdm(it, desc=tqdm_desc, total=n)
        return iter(self.fetch_batch(i) for i in it)


class ITimeSeriesProcessor(IMLDataProcessor):
    config: TimeSeriesConfig

    _train_bundle: TimeSeriesDataBundle
    _validation_bundle: Optional[TimeSeriesDataBundle]

    config_base = TimeSeriesConfig
    cache_paths_base = TimeSeriesCachePaths

    # abstract

    @property
    @abstractmethod
    def tag(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def prepare_data(cls, config: TimeSeriesConfig) -> np.ndarray:
        pass

    @abstractmethod
    def get_time_anchors(self, times: np.ndarray) -> np.ndarray:
        """
        return the time anchors based on `data[time_columns]`, which will be used to
        calculate the sorted indices
        """

    @abstractmethod
    def check_valid(self, mat: np.ndarray) -> bool:
        """check whether `mat` is a valid data sample"""

    def sanity_check(self, tag: MLDatasetTag, bundle: TimeSeriesDataBundle) -> None:
        def check_ids(data: np.ndarray) -> np.ndarray:
            ids_ = data[..., self.config.id_column].astype(int)
            if np.abs(ids_[:, 1:] - ids_[:, :-1]).sum().item() != 0:
                raise ValueError(f"multiple ids occurred in one sample : {ids_}")
            return ids_

        def get_time_anchors(data: np.ndarray) -> np.ndarray:
            times = data[..., self.config.time_columns]
            time_anchors = self.get_time_anchors(times.reshape([-1, times.shape[-1]]))
            time_anchors = time_anchors.reshape(times.shape[:-1])
            return time_anchors

        min_max_x_anchor = None
        min_max_y_anchor = None
        max_max_x_anchor = None
        max_max_y_anchor = None
        id_counts: Counter = Counter()
        loader = bundle.to_loader(batch_size=512, tqdm_desc="check_split")
        for batch in loader:
            x, y = batch.input, batch.labels
            full = x if y is None or batch.others is None else batch.others["full"]
            ids = check_ids(full)
            x_anchors = get_time_anchors(x)
            y_anchors = None if y is None else get_time_anchors(y)
            max_x_anchors = x_anchors.max(axis=-1)
            max_y_anchors = None if y_anchors is None else y_anchors.max(axis=-1)
            if tag == MLDatasetTag.TRAIN:
                if not self.config.for_inference:
                    if self.config.validation_split is not None:
                        anchors = x_anchors if y_anchors is None else y_anchors
                        max_anchors = (
                            max_x_anchors if max_y_anchors is None else max_y_anchors
                        )
                        if np.any(max_anchors >= self.config.validation_split):
                            msg = f"train data exceeds validation split : {anchors}"
                            raise ValueError(msg)
                else:
                    if self.config.num_test is not None:
                        for id_ in ids[:, 0]:
                            if id_counts[id_] >= self.config.num_test:
                                raise ValueError("test data exceeds num_test")
                            id_counts[id_] += 1
                    elif self.config.test_split is not None:
                        if np.any(max_x_anchors < self.config.test_split):
                            msg = f"test data exceeds test split : {x_anchors}"
                            raise ValueError(msg)
                        if self.config.test_end is not None and np.any(
                            max_x_anchors >= self.config.test_end
                        ):
                            msg = f"test data exceeds test end : {x_anchors}"
                            raise ValueError(msg)
            elif tag == MLDatasetTag.VALID:
                if self.config.validation_split is not None and np.any(
                    max_x_anchors < self.config.validation_split
                ):
                    raise ValueError(
                        f"validation data exceeds validation split : {x_anchors}"
                    )
                if self.config.validation_end is not None:
                    anchors = x_anchors if y_anchors is None else y_anchors
                    max_anchors = (
                        max_x_anchors if max_y_anchors is None else max_y_anchors
                    )
                    if np.any(max_anchors >= self.config.validation_end):
                        msg = f"validation data exceeds validation end : {anchors}"
                        raise ValueError(msg)
            # update statics
            if min_max_x_anchor is None:
                min_max_x_anchor = max_x_anchors.min().item()
            else:
                min_max_x_anchor = min(max_x_anchors.min().item(), min_max_x_anchor)
            if max_max_x_anchor is None:
                max_max_x_anchor = max_x_anchors.max().item()
            else:
                max_max_x_anchor = max(max_x_anchors.max().item(), max_max_x_anchor)
            if max_y_anchors is not None:
                if min_max_y_anchor is None:
                    min_max_y_anchor = max_y_anchors.min().item()
                else:
                    min_max_y_anchor = min(max_y_anchors.min().item(), min_max_y_anchor)
            if max_y_anchors is not None:
                if max_max_y_anchor is None:
                    max_max_y_anchor = max_y_anchors.max().item()
                else:
                    max_max_y_anchor = max(max_y_anchors.max().item(), max_max_y_anchor)
        print(f"> min max_x_anchor of {tag} : {min_max_x_anchor}")
        print(f"> min max_y_anchor of {tag} : {min_max_y_anchor}")
        print(f"> max max_x_anchor of {tag} : {max_max_x_anchor}")
        print(f"> max max_y_anchor of {tag} : {max_max_y_anchor}")

    # utils

    def verbose(self, msg: str) -> None:
        if self.config.verbose:
            print_info(msg)

    @property
    def hashed_cache_folder(self) -> str:
        return os.path.join(self.config.cache_folder, self.tag, self.config.hash)

    def get_cache_paths(self, tag: MLDatasetTag) -> TimeSeriesCachePaths:
        folder = os.path.join(self.hashed_cache_folder, tag)
        paths = dict(
            data_path=os.path.join(folder, "data.npy"),
            merged_indices_path=os.path.join(folder, "merged_indices.npy"),
            split_indices_path=os.path.join(folder, "split_indices.json"),
            valid_indices_path=os.path.join(folder, "valid_indices.npy"),
        )
        paths = {k: os.path.abspath(v) for k, v in paths.items()}
        return TimeSeriesCachePaths(**paths)

    def load_cache_paths(self, paths: TimeSeriesCachePaths) -> TimeSeriesDataBundle:
        self.verbose("loading data")
        data = np.load(paths.data_path)
        self.verbose("loading rolled indices")
        np_merged_indices = np.load(paths.merged_indices_path)
        rolled_indices = StrideArray(np_merged_indices).roll(self.config.span)
        self.verbose("loading split indices")
        with open(paths.split_indices_path, "r") as f:
            split_indices = json.load(f)
        self.verbose("loading valid indices")
        valid_indices = np.load(paths.valid_indices_path)
        # return
        args = self.config.copy(), data, split_indices, rolled_indices, valid_indices
        return TimeSeriesDataBundle(*args)

    def _get_valid_indices(
        self,
        i: int,
        num: int,
        data: np.ndarray,
        rolled_indices: np.ndarray,
        num_indices_cumsum: np.ndarray,
        boolean_mask: Optional[np.ndarray],
    ) -> Optional[List[int]]:
        if num < self.config.span:
            return [] if boolean_mask is None else None
        cumulate = 0 if i == 0 else num_indices_cumsum[i - 1]
        valid_indices: Optional[List[int]] = [] if boolean_mask is None else None
        for j in range(num - self.config.span + 1):
            idx = j + cumulate
            is_valid = False
            if self.config.num_test is not None and self.config.enforce_test_valid:
                is_valid = True
            elif self.check_valid(data[rolled_indices[idx]]):
                is_valid = True
            if is_valid:
                if boolean_mask is not None:
                    boolean_mask[idx] = True
                elif valid_indices is not None:
                    valid_indices.append(idx)
        return valid_indices

    def get_bundle(
        self,
        x_train: np.ndarray,
        indices: np.ndarray,
        cache_paths: TimeSeriesCachePaths,
    ) -> TimeSeriesDataBundle:
        # data
        data = x_train[indices]
        if not self.config.no_cache:
            np.save(cache_paths.data_path, data)
        # split indices
        self.verbose("group by id")
        ids = data[..., self.config.id_column]
        split_indices = get_unique_indices(ids).split_indices
        split_indices = [indices.tolist() for indices in split_indices]
        if self.config.num_test is not None and self.config.enforce_num_test:
            span = self.config.x_window + self.config.num_test - 1
            split_indices = [indices[-span:] for indices in split_indices]
        if not self.config.no_cache:
            with open(cache_paths.split_indices_path, "w") as f:
                json.dump(split_indices, f)
        # rolled indices
        merged_indices = []
        for indices in split_indices:
            merged_indices += indices
        np_merged_indices = np.array(merged_indices, int)
        rolled_indices = StrideArray(np_merged_indices).roll(self.config.span)
        if not self.config.no_cache:
            np.save(cache_paths.merged_indices_path, np_merged_indices)
        # valid indices
        fn = self._get_valid_indices
        num_indices = np.array(list(map(len, split_indices)), int)
        num_indices_cumsum = np.cumsum(num_indices)
        if not self.config.num_workers:
            valid_indices_list: List[int] = []
            for i, num in enumerate(tqdm(num_indices, desc="check_valid")):
                i_args = i, num, data, rolled_indices, num_indices_cumsum, None
                valid_indices_list += fn(*i_args) or []
            valid_indices = np.array(valid_indices_list, int)
        else:
            self.verbose("preparing shared arrays")
            shared_arrays = (
                SharedArray.from_data(data),
                SharedArray.from_data(np_merged_indices),
                SharedArray.from_data(num_indices_cumsum),
                SharedArray(np.bool, [num_indices_cumsum[-1]]),
            )
            common_args = [shared.value for shared in shared_arrays]
            common_args[1] = StrideArray(common_args[1]).roll(self.config.span)
            args_list: List[List[Any]] = [[] for _ in range(6)]
            for i, num in enumerate(num_indices):
                for j, arg in enumerate([i, num, *common_args]):
                    args_list[j].append(arg)
            self.verbose("launching workers")
            Parallel(self.config.num_workers).grouped(fn, *args_list)
            valid_indices = np.nonzero(common_args[-1])[0]
            for shared in shared_arrays:
                shared.destroy()
        if not self.config.no_cache:
            np.save(cache_paths.valid_indices_path, valid_indices)
        # random sample
        random_ratio = self.config.random_sample_ratio
        if not self.config.for_inference and random_ratio is not None:
            n = len(valid_indices)
            sampled = np.random.permutation(n)[: round(n * random_ratio)]
            valid_indices = valid_indices[sampled]
        # return
        args = self.config.copy(), data, split_indices, rolled_indices, valid_indices
        return TimeSeriesDataBundle(*args)

    def get_split(
        self,
        split: int,
        end: Optional[int],
        sorted_indices: np.ndarray,
        sorted_anchors: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # `split` belongs to `right_indices`
        unique_anchors = np.unique(sorted_anchors)
        right_mask = unique_anchors >= split
        right_end_mask = None
        if end is not None:
            right_end_mask = unique_anchors < end
            right_mask = right_mask & right_end_mask
        y_span = self.config.gap + self.config.y_window
        if right_mask.sum().item() <= y_span:
            if self.config.verbose:
                print_warning(
                    "validation data is not enough when "
                    f"`validation_split` is set to `{split}`"
                    f"& `validation_end` is set to `{end}`"
                )
            return sorted_indices, None
        left_idx = np.argmax(right_mask).item() - 1
        left_anchor = unique_anchors[left_idx]
        left_mask = sorted_anchors <= left_anchor
        right_idx = max(0, left_idx - self.config.x_window + 2)
        right_anchor = unique_anchors[right_idx]
        right_mask = sorted_anchors >= right_anchor
        if right_end_mask is not None:
            right_end_idx = np.argmin(right_end_mask) - 1
            right_end_idx = min(len(unique_anchors) - 1, right_end_idx.item())
            right_end_anchor = unique_anchors[right_end_idx]
            right_mask = right_mask & (sorted_anchors <= right_end_anchor)
        left_indices = sorted_indices[left_mask]
        right_indices = sorted_indices[right_mask]
        return left_indices, right_indices

    # inheritance

    def build_with(self, config: TimeSeriesConfig, x_train: np.ndarray) -> None:  # type: ignore
        self.config = config
        if not config.no_cache:
            os.makedirs(self.hashed_cache_folder, exist_ok=True)
            with open(os.path.join(self.hashed_cache_folder, "config.json"), "w") as f:
                json.dump(config.asdict(), f)
        # cache
        tr_paths = self.get_cache_paths(MLDatasetTag.TRAIN)
        cv_paths = self.get_cache_paths(MLDatasetTag.VALID)
        if not config.no_cache:
            os.makedirs(tr_paths.folder, exist_ok=True)
            if config.use_validation:
                os.makedirs(cv_paths.folder, exist_ok=True)
        # load
        if (
            not config.no_cache
            and tr_paths.all_exists
            and (not config.use_validation or cv_paths.all_exists)
        ):
            self.verbose("loading train caches")
            self._train_bundle = self.load_cache_paths(tr_paths)
            if not config.use_validation:
                self._validation_bundle = None
            else:
                self.verbose("loading validation caches")
                self._validation_bundle = self.load_cache_paths(cv_paths)
        # build
        else:
            self.verbose("sort by time")
            time_columns = x_train[..., config.time_columns].astype(int)
            time_anchors = self.get_time_anchors(time_columns).ravel()
            sorted_indices = np.argsort(time_anchors)
            if config.use_validation:
                sorted_anchors = time_anchors[sorted_indices]
                tr_indices, cv_indices = self.get_split(
                    config.validation_split,  # type: ignore
                    config.validation_end,
                    sorted_indices,
                    sorted_anchors,
                )
                self.verbose("generating train bundle")
                train_bundle = self.get_bundle(x_train, tr_indices, tr_paths)
                if cv_indices is None:
                    validation_bundle = None
                else:
                    self.verbose("generating validation bundle")
                    validation_bundle = self.get_bundle(x_train, cv_indices, cv_paths)
                self._train_bundle = train_bundle
                self._validation_bundle = validation_bundle
            else:
                tag = "test" if config.for_inference else "train"
                self.verbose(f"generating {tag} bundle")
                if not config.for_inference:
                    indices = sorted_indices
                else:
                    if config.num_test is not None and config.test_end is not None:
                        msg = "should not set `test_end` when `num_test` is set"
                        raise ValueError(msg)
                    if config.num_test is None and config.test_split is None:
                        indices = sorted_indices
                    elif config.num_test is not None and config.enforce_num_test:
                        indices = sorted_indices
                    else:
                        sorted_anchors = time_anchors[sorted_indices]
                        if config.test_split is not None:
                            test_split = config.test_split
                        else:
                            test_split = sorted_anchors[-config.num_test]  # type: ignore
                        _, indices = self.get_split(
                            test_split,
                            config.test_end,
                            sorted_indices,
                            sorted_anchors,
                        )
                self._train_bundle = self.get_bundle(x_train, indices, tr_paths)
                self._validation_bundle = None
        self.verbose("done")

    def preprocess(self, config: TimeSeriesConfig) -> IMLPreProcessedData:  # type: ignore
        num_test_enforced = config.num_test is not None and config.enforce_test_valid
        if config.sanity_check and not num_test_enforced:
            self.sanity_check(MLDatasetTag.TRAIN, self._train_bundle)
            if self._validation_bundle is not None:
                self.sanity_check(MLDatasetTag.VALID, self._validation_bundle)
        if self._validation_bundle is None:
            x_valid = None
        else:
            x_valid = self._validation_bundle.data
        return IMLPreProcessedData(
            self._train_bundle.data,
            x_valid=x_valid,
            num_history=self.config.x_window,
            is_classification=False,
        )

    def dumps(self) -> Dict[str, Any]:
        d = self.config.asdict()
        d["tr_cache_paths"] = self.get_cache_paths(MLDatasetTag.TRAIN).asdict()
        if not self.config.use_validation:
            d["cv_cache_paths"] = None
        else:
            d["cv_cache_paths"] = self.get_cache_paths(MLDatasetTag.VALID).asdict()
        return d

    def loads(self, dumped: Dict[str, Any]) -> None:
        def _load(d: Dict[str, Any], tag: str) -> TimeSeriesDataBundle:
            self.verbose(f"loading {tag} caches")
            cache_paths = self.cache_paths_base(**d)
            return self.load_cache_paths(cache_paths)

        tr_paths_d = dumped.pop("tr_cache_paths")
        cv_paths_d = dumped.pop("cv_cache_paths")
        self.config = self.config_base(**dumped)
        self.verbose("load begin")
        self._train_bundle = _load(tr_paths_d, "train")
        if cv_paths_d is None:
            self._validation_bundle = None
        else:
            self._validation_bundle = _load(cv_paths_d, "validation")
        self.verbose("load complete")

    # callbacks

    def get_num_samples(self, x: np.ndarray, tag: MLDatasetTag) -> int:
        if tag == MLDatasetTag.TRAIN:
            return len(self._train_bundle)
        if self._validation_bundle is None:
            raise ValueError(
                "`_validation_bundle` is not ready "
                "but `get_num_samples` with `tag=valid` is called"
            )
        return len(self._validation_bundle)

    def fetch_batch(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray],
        indices: Union[int, List[int], np.ndarray],
        tag: MLDatasetTag,
    ) -> IMLBatch:
        if tag == MLDatasetTag.TRAIN:
            bundle = self._train_bundle
        else:
            if self._validation_bundle is None:
                raise ValueError(
                    "`_validation_bundle` is not ready "
                    "but `fetch_batch` with `tag=valid` is called"
                )
            bundle = self._validation_bundle
        return bundle.fetch_batch(indices)

    # api

    def get_flatten(self, indices: np.ndarray, tag: MLDatasetTag) -> np.ndarray:
        if tag == MLDatasetTag.TRAIN:
            bundle = self._train_bundle
        else:
            if self._validation_bundle is None:
                raise ValueError(
                    "`_validation_bundle` is not ready "
                    "but `get_flatten` with `tag=valid` is called"
                )
            bundle = self._validation_bundle
        indices = bundle.rolled_indices[bundle.valid_indices[indices]][..., 0]
        return bundle.data[indices]


@register_ml_data(TIME_SERIES_NAME)
class TimeSeriesData(IMLData):
    processor: ITimeSeriesProcessor
    processor_type = TIME_SERIES_NAME

    def __init__(
        self,
        ts_config: TimeSeriesConfig,
        *,
        data: Optional[np.ndarray] = None,
        processor: Optional[IMLDataProcessor] = None,
        shuffle_train: bool = True,
        shuffle_valid: bool = False,
        batch_size: int = 128,
        valid_batch_size: int = 512,
        use_numpy: bool = False,
    ):
        self.ts_config = ts_config
        if data is None:
            processor_cls = ITimeSeriesProcessor.get(self.processor_type)
            data = processor_cls.prepare_data(ts_config)
        super().__init__(
            data,
            processor=processor,
            shuffle_train=shuffle_train,
            shuffle_valid=shuffle_valid,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            use_numpy=use_numpy,
            for_inference=ts_config.for_inference,
        )

    @property
    def train_bundle(self) -> TimeSeriesDataBundle:
        return self.processor._train_bundle

    @property
    def validation_bundle(self) -> Optional[TimeSeriesDataBundle]:
        return self.processor._validation_bundle

    @property
    def processor_build_config(self) -> TimeSeriesConfig:
        return self.ts_config

    @property
    def processor_preprocess_config(self) -> TimeSeriesConfig:
        return self.ts_config


def make_ts_test_data(
    config: TimeSeriesConfig,
    *,
    data: Optional[np.ndarray] = None,
    test_split: Optional[int] = None,
    test_end: Optional[int] = None,
    num_test: Optional[int] = None,
    enforce_num_test: bool = False,
    sanity_check: bool = True,
    use_numpy: bool = False,
    no_cache: bool = True,
    batch_size: int = 1024,
) -> TimeSeriesData:
    config = config.copy()
    config.for_inference = True
    config.validation_split = None
    config.validation_end = None
    config.test_split = test_split
    config.test_end = test_end
    config.num_test = num_test
    config.enforce_num_test = enforce_num_test
    config.sanity_check = sanity_check
    config.no_cache = no_cache
    config.y_window = 0
    config.gap = 0
    test_data = TimeSeriesData(
        config,
        data=data,
        batch_size=batch_size,
        shuffle_train=False,
        use_numpy=use_numpy,
    )
    test_data.prepare(None)
    return test_data


class ITSRollingInferencePostProcess(Protocol):
    def __call__(self, i: int, predictions: np.ndarray, raw_data: np.ndarray) -> None:
        """we need to fill `predictions` to `raw_data` to perform rolling inference"""


def ts_rolling_inference(
    m: Any,
    data: np.ndarray,
    config: TimeSeriesConfig,
    steps: int,
    *,
    num_test: Optional[int] = None,
    enforce_num_test: bool = False,
    sanity_check: bool = True,
    use_numpy: bool = False,
    no_cache: bool = True,
    batch_size: int = 1024,
    postprocess: Optional[ITSRollingInferencePostProcess] = None,
) -> np.ndarray:
    results = None
    for i in range(steps):
        print_info(f"rolling inference {i+1} / {steps}")
        if results is None:
            test_arr = data
        else:
            test_arr = np.vstack([data, results])
        test_data = make_ts_test_data(
            config.copy(),
            data=test_arr,
            num_test=num_test,
            enforce_num_test=enforce_num_test,
            sanity_check=sanity_check,
            use_numpy=use_numpy,
            no_cache=no_cache,
            batch_size=batch_size,
        )
        i_predictions = m.predict(test_data)["predictions"]
        i_raw_data = (
            test_data.processor.get_flatten(
                np.array(list(range(len(i_predictions)))),
                MLDatasetTag.TRAIN,
            )
            .copy()
            .astype(np.float64)
        )
        if postprocess is not None:
            postprocess(i, i_predictions, i_raw_data)
        print_info(f"filled data ({i+1}) {i_raw_data.shape}")
        if results is None:
            results = i_raw_data
        else:
            results = np.vstack([results, i_raw_data])
    return results


def register_ts_processor() -> Callable:
    return register_ml_data_processor(TIME_SERIES_NAME)


__all__ = [
    "make_ts_test_data",
    "ts_rolling_inference",
    "register_ts_processor",
    "TimeSeriesConfig",
    "ITimeSeriesProcessor",
    "TimeSeriesData",
]
