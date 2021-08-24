import os
import math
import logging
import warnings

import numba as nb
import numpy as np

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Optional
from typing import NamedTuple
from cftool.misc import hash_code
from cftool.misc import fix_float_to_length
from cftool.misc import StrideArray
from cftool.misc import LoggingMixin
from cftool.stat import RollingStat

from .data import get_weighted_indices
from .data import MLDataset
from ...types import tensor_dict_type
from ...protocol import DataLoaderProtocol
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY
from ...constants import BATCH_INDICES_KEY
from ...misc.toolkit import squeeze
from ...misc.toolkit import to_torch
from ...misc.toolkit import WithRegister
from ...misc.toolkit import SharedArrayWrapper


class DataConfig(NamedTuple):
    x_window: int
    y_window: int  # y_window could be 0, which means it will use x_window as its value
    nan_ratio: float
    gap: int
    train_split_tick: str
    train_start_tick: Optional[str] = None
    test_split_tick: Optional[str] = None
    test_end_tick: Optional[str] = None
    use_target_as_input: bool = True
    sample_by_ticks: bool = False
    nan_fill: Union[str, float] = 0.0
    callback: str = "batch_first_prod"
    callback_config: Optional[Dict[str, Any]] = None
    filter_fns: Optional[Dict[str, str]] = None
    filter_fns_config: Optional[Dict[str, Dict[str, Any]]] = None
    xy_padding: int = 1
    data_cache_folder: str = ".data_cache"
    use_memory_sharing: bool = True

    @property
    def y_span(self) -> int:
        return self.y_window + self.xy_padding


filter_fns: Dict[str, Type["FilterFunctions"]] = {}
data_callbacks: Dict[str, Type["DataCallback"]] = {}


class DataCallback(WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type["DataCallback"]] = data_callbacks

    def __init__(self, cfg: DataConfig, is_clf: bool):
        self.cfg = cfg
        self.is_clf = is_clf
        self.config = cfg.callback_config or {}

    @abstractmethod
    def process_batch(self, x: np.ndarray, labels: np.ndarray) -> tensor_dict_type:
        pass

    @staticmethod
    def _to_dict(x: np.ndarray, labels: np.ndarray) -> tensor_dict_type:
        return dict(zip([INPUT_KEY, LABEL_KEY], map(to_torch, [x, labels])))

    @classmethod
    def make_with(cls, cfg: DataConfig, is_clf: bool) -> "DataCallback":
        return data_callbacks[cfg.callback](cfg, is_clf)


@DataCallback.register("identity")
class Identity(DataCallback):
    def process_batch(self, x: np.ndarray, labels: np.ndarray) -> tensor_dict_type:
        return self._to_dict(x, labels)


class BatchFirstAggregate(DataCallback, metaclass=ABCMeta):
    def __init__(self, cfg: DataConfig, is_clf: bool):
        super().__init__(cfg, is_clf)
        self.clf_threshold = self.config.setdefault("clf_threshold", 0.0015)
        self.label_multiplier = self.config.setdefault("label_multiplier", None)

    @abstractmethod
    def aggregate_labels(self, labels: np.ndarray) -> np.ndarray:
        pass

    def process_batch(self, x: np.ndarray, labels: np.ndarray) -> tensor_dict_type:
        if labels.shape[1] > 1:
            labels = self.aggregate_labels(labels)
        if self.is_clf:
            labels = (labels > self.clf_threshold).astype(np.int64)
        elif self.label_multiplier is not None:
            labels *= self.label_multiplier
        return self._to_dict(x, labels)


@DataCallback.register("batch_first_prod")
class BatchFirstProd(BatchFirstAggregate):
    def aggregate_labels(self, labels: np.ndarray) -> np.ndarray:
        return np.prod(labels + 1.0, axis=-1, keepdims=True) - 1.0


@DataCallback.register("batch_first_cum_prod")
class BatchFirstCumProd(BatchFirstAggregate):
    def aggregate_labels(self, labels: np.ndarray) -> np.ndarray:
        return np.cumprod(labels + 1.0, axis=-1) - 1.0


class FilterFunctions(WithRegister):
    d: Dict[str, Type["FilterFunctions"]] = filter_fns

    def __init__(self, key: str, cfg: DataConfig, is_clf: bool):
        self.key = key
        self.cfg = cfg
        self.is_clf = is_clf
        self.config = (cfg.filter_fns_config or {}).get(key, {})

    def get_invalid_tick_mask(self, ticks: np.ndarray) -> np.ndarray:
        pass

    def get_invalid_objects_mask(self, objects: List[str]) -> np.ndarray:
        pass

    def get_valid_labels_mask(self, labels: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def make_with(cls, cfg: DataConfig, is_clf: bool) -> Dict[str, "FilterFunctions"]:
        if cfg.filter_fns is None:
            raise ValueError("`filter_fns` is not provided")
        return {k: filter_fns[v](k, cfg, is_clf) for k, v in cfg.filter_fns.items()}


@FilterFunctions.register("keep_edge")
class KeepEdge(FilterFunctions):
    def __init__(self, key: str, cfg: DataConfig, is_clf: bool):
        super().__init__(key, cfg, is_clf)
        data_callback = DataCallback.make_with(cfg, is_clf)
        assert isinstance(data_callback, BatchFirstAggregate)
        self.data_callback = data_callback

    def get_valid_labels_mask(self, labels: np.ndarray) -> np.ndarray:
        labels = np.transpose(labels, [0, 2, 1])
        ratio = self.config.setdefault("ratio", 0.1)
        aggregated = squeeze(self.data_callback.aggregate_labels(labels))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            floor = np.nanquantile(aggregated, ratio, axis=1, keepdims=True)
            ceiling = np.nanquantile(aggregated, 1.0 - ratio, axis=1, keepdims=True)
        return (aggregated <= floor) | (aggregated >= ceiling)


@nb.jit(nopython=True, parallel=True)
def rolling_valid(x: np.ndarray, window: int) -> np.ndarray:
    nb.set_num_threads(8)
    out = np.full(x.shape[:2], True)
    for i in nb.prange(out.shape[0]):
        for j in range(out.shape[1]):
            if np.any(x[i, j]):
                out[i, j : j + window] = False
    return out[..., window - 1 :]


def pad_head(arr: np.ndarray, pad_width: int, pad_value: Any) -> np.ndarray:
    return np.pad(
        arr,
        pad_width=[[pad_width, 0], [0, 0]],
        mode="constant",
        constant_values=pad_value,
    )


def get_valid_mask(arr: np.ndarray, window: int, nan_ratio: float) -> np.ndarray:
    nan_mask = np.isnan(arr)
    if nan_ratio <= 0.0:
        all_valid = rolling_valid(nan_mask.transpose([1, 0, 2]), window).T
    else:
        nan_mask = nan_mask.astype(np.float32)
        nan_ratio_mat = RollingStat.mean(nan_mask, window, axis=0)
        all_valid = np.all(nan_ratio_mat <= nan_ratio, axis=-1)
    return pad_head(all_valid, window - 1, False)


class TSBundle(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    ticks: np.ndarray
    indices: np.ndarray
    offset: int


class SpanSplit(NamedTuple):
    train_span: np.ndarray
    valid_span: np.ndarray
    test_span: Optional[np.ndarray] = None


class TimeSeriesDataManager(LoggingMixin):
    data: MLDataset

    def __init__(
        self,
        cfg: DataConfig,
        code: str,
        *,
        is_clf: bool,
        cache_folder: str = os.path.join(os.path.expanduser("~"), ".cflearn", "ts"),
    ):
        self.cfg = cfg
        self.code = code
        self.is_clf = is_clf
        self.cache_folder = cache_folder

        # saw
        def _get_saw(file: str) -> SharedArrayWrapper:
            to_memory = cfg.use_memory_sharing
            return SharedArrayWrapper(code_folder, file, to_memory=to_memory)

        code_folder = os.path.join(cache_folder, code)
        os.makedirs(code_folder, exist_ok=True)
        self.x_saw = _get_saw("x.npy")
        self.y_saw = _get_saw("y.npy")
        self.ticks_saw = _get_saw("ticks.npy")
        nan_ratio_str = fix_float_to_length(cfg.nan_ratio, 6)
        window_str = f"{cfg.x_window}-{cfg.y_window}"
        filter_str = f"{cfg.filter_fns}_{cfg.filter_fns_config}"
        valid_mask_code = hash_code(f"{nan_ratio_str}{window_str}{filter_str}")
        self.valid_mask_saw = _get_saw(f"valid_mask_{valid_mask_code}.npy")

    @property
    def is_ready(self) -> bool:
        if not self.x_saw.is_ready:
            return False
        if not self.y_saw.is_ready:
            return False
        if not self.valid_mask_saw.is_ready:
            return False
        return True

    def _get_span(self, ticks: np.ndarray, mask: np.ndarray, gap: int) -> np.ndarray:
        indices = np.nonzero(mask)[0]
        if gap > 0:
            indices = indices[gap:]
        if indices.shape[0] <= 0:
            raise ValueError(
                "tick settings are invalid, "
                "which results in no valid data are available"
            )
        # pad first
        first_index = indices[0]
        padded_indices = first_index - np.arange(self.cfg.x_window - 1)[::-1] - 1
        padded_indices = padded_indices[padded_indices >= 0]
        indices = np.append(padded_indices, indices)
        # pad last
        if self.cfg.y_window > 0:
            last_index = indices[-1]
            padded_indices = last_index + np.arange(self.cfg.y_span) + 1
            padded_indices = padded_indices[padded_indices < len(ticks)]
            indices = np.append(indices, padded_indices)
        return indices[[0, -1]]

    def make_data(
        self,
        x: Optional[np.ndarray],
        y: Optional[np.ndarray],
        ticks: Optional[np.ndarray],
    ) -> None:
        # sanity check
        if x is None:
            if not self.x_saw.is_ready:
                raise ValueError("`x` should be provided because it is not ready")
            x = self.x_saw.read()
        if y is None:
            if not self.y_saw.is_ready:
                raise ValueError("`y` should be provided because it is not ready")
            y = self.y_saw.read()
        if ticks is None:
            if not self.ticks_saw.is_ready:
                raise ValueError("`ticks` should be provided because it is not ready")
            ticks = self.ticks_saw.read()
        if len(x.shape) != 3:
            raise ValueError("`x` should be 3D array (ticks, objects, dim)")
        if len(x) != len(y):
            raise ValueError("length of `x` should equal to length of `y`")
        if len(x) != len(ticks):
            raise ValueError("length of `x` should equal to length of `ticks`")
        # filters
        if self.cfg.filter_fns is None:
            filters = None
        else:
            filters = FilterFunctions.make_with(self.cfg, self.is_clf)
        # valid mask
        x_window = self.cfg.x_window
        y_window = self.cfg.y_window or x_window
        x_mask = get_valid_mask(x, x_window, self.cfg.nan_ratio)
        y_mask = get_valid_mask(y[..., None], y_window, self.cfg.nan_ratio)
        if filters is not None:
            # for x & y, only train dataset needs to be filtered
            label_filter_fn = filters.get("label")
            if y_mask is not None and label_filter_fn is not None:
                train_split = self.cfg.train_split_tick
                num_train = (ticks <= train_split).sum()
                if self.cfg.y_window > 0:
                    num_train += self.cfg.y_span
                train_target = x[:num_train]
                rolled_target = StrideArray(train_target).roll(y_window, axis=0)
                filter_y_mask = label_filter_fn.get_valid_labels_mask(rolled_target)
                y_mask[y_window - 1 : num_train] &= filter_y_mask
            # for ticks & objects, all datasets need to be filtered
            tick_filter_fn = filters.get("tick")
            if tick_filter_fn is not None:
                tick_invalid_mask = tick_filter_fn.get_invalid_tick_mask(ticks)
                x_mask[tick_invalid_mask] = False
        if self.cfg.y_window == 0:
            valid_mask = x_mask & y_mask
        else:
            offset = x_window + self.cfg.y_span - 1
            offset_valid = np.zeros_like(y_mask, np.bool_)
            offset_valid[x_window - 1 : -self.cfg.y_span] = y_mask[offset:]
            valid_mask = x_mask & offset_valid
        # sanity check
        invalid_indices = np.nonzero(~np.any(valid_mask, 1))[0][x_window:]
        if self.cfg.y_window > 0:
            invalid_indices = invalid_indices[: -self.cfg.y_span]
        if invalid_indices.shape[0] != 0 and self.cfg.filter_fns is None:
            self.log_msg(
                "invalid data occurred\n"
                f"* ticks   : {ticks[invalid_indices].tolist()}\n"
                f"* indices : {invalid_indices.tolist()}",
                self.warning_prefix,
                msg_level=logging.WARNING,
            )
        # finalize
        self.x_saw.write(x)
        self.y_saw.write(y)
        self.ticks_saw.write(ticks)
        self.valid_mask_saw.write(valid_mask)

    def split(self) -> SpanSplit:
        if not self.is_ready:
            raise ValueError("`TimeSeriesDataManager` is not ready yet")
        ticks = self.ticks_saw.read()
        # (train_start -> ) train_split -> test_split -> test_end
        if (
            self.cfg.test_split_tick is None
            or self.cfg.test_split_tick > self.cfg.train_split_tick
        ):
            train_mask = ticks <= self.cfg.train_split_tick
            remained_mask = ~train_mask
            if self.cfg.train_start_tick is not None:
                train_mask &= ticks >= self.cfg.train_start_tick
            if self.cfg.test_split_tick is None:
                validation_mask = test_mask = None
                if self.cfg.test_end_tick is not None:
                    remained_mask &= ticks <= self.cfg.test_end_tick
            else:
                test_mask = ticks > self.cfg.test_split_tick
                validation_mask = remained_mask & ~test_mask
                if self.cfg.test_end_tick is not None:
                    test_mask &= ticks <= self.cfg.test_end_tick
        # (train_start -> ) test_split -> train_split -> test_end
        else:
            validation_mask = ticks <= self.cfg.test_split_tick
            remained_mask = ~validation_mask
            if self.cfg.train_start_tick is not None:
                validation_mask &= ticks >= self.cfg.train_start_tick
            test_mask = ticks > self.cfg.train_split_tick
            train_mask = remained_mask & ~test_mask
            if self.cfg.test_end_tick is not None:
                test_mask &= ticks <= self.cfg.test_end_tick
        # train
        train_span = self._get_span(ticks, train_mask, 0)
        # valid
        if self.cfg.test_split_tick is None:
            valid_span = self._get_span(ticks, remained_mask, self.cfg.gap)
            return SpanSplit(train_span, valid_span)
        assert validation_mask is not None
        valid_span = self._get_span(ticks, validation_mask, self.cfg.gap)
        # test
        assert test_mask is not None
        test_span = self._get_span(ticks, test_mask, self.cfg.gap)
        return SpanSplit(train_span, valid_span, test_span)

    def pick_from_span(
        self,
        span: np.ndarray,
        masks: Optional[List[np.ndarray]] = None,
    ) -> TSBundle:
        # sanity check
        x, y = self.x_saw.read(), self.y_saw.read()
        if not self.is_ready:
            raise ValueError("`TimeSeriesDataManager` is not ready yet")
        if masks is not None:
            for mask in masks:
                if mask.shape != x.shape[:2]:
                    raise ValueError("`masks` is not compatible with `x`")
        # pick
        begin, end = span.tolist()
        # x & y
        x, y = x[begin:end], y[begin:end]
        # ticks
        ticks = self.ticks_saw.read()[begin:end]
        # indices
        valid_mask = self.valid_mask_saw.read().copy()
        if masks is not None:
            offset_start = self.cfg.x_window - 1
            padding = self.cfg.xy_padding + 1
            offset = offset_start + padding
            for mask in masks:
                local_valid = np.zeros_like(mask, np.bool_)
                local_valid[offset_start:-padding] = mask[offset:]
                valid_mask &= local_valid
        valid_mask[begin : begin + self.cfg.x_window - 1] = False
        if self.cfg.y_window > 0:
            valid_mask[end - self.cfg.y_span : end] = False
        tick_indices, object_indices = np.nonzero(valid_mask)
        indices = np.vstack([tick_indices, object_indices]).T
        return TSBundle(x, y, ticks, indices, begin)


@DataLoaderProtocol.register("ts")
class TimeSeriesLoader(DataLoaderProtocol):
    data: MLDataset

    def __init__(
        self,
        cfg: DataConfig,
        bundle: TSBundle,
        *,
        is_clf: bool,
        shuffle: bool,
        batch_size: int = 128,
        sample_weights: Optional[np.ndarray] = None,
    ):
        if sample_weights is not None and len(bundle.x) != len(sample_weights):
            raise ValueError(
                f"the number of data samples ({len(bundle.x)}) is not identical with "
                f"the number of sample weights ({len(sample_weights)})"
            )
        super().__init__(sample_weights=sample_weights)
        self.name = None
        self.cfg = cfg
        self.bundle = bundle
        self.is_clf = is_clf
        self.shuffle = shuffle
        self.shuffle_backup = shuffle
        self.batch_size = batch_size
        self.callback = DataCallback.make_with(cfg, is_clf)
        self.indices = self.bundle.indices.copy() - self.bundle.offset
        # prepare data
        self.num_ticks, self.num_objects, self.dim = bundle.x.shape
        x_window = cfg.x_window
        y_window = cfg.y_window
        rolled_x = StrideArray(bundle.x).roll(x_window, axis=0)
        rolled_y = StrideArray(bundle.y).roll(y_window or x_window, axis=0)
        self.data = MLDataset(rolled_x, rolled_y)

    def __len__(self) -> int:
        return math.ceil(self.bundle.indices.shape[0] / self.batch_size)

    def __iter__(self) -> "TimeSeriesLoader":
        self._cursor = -1
        indices = self.indices.copy()
        if self.shuffle:
            weighted = get_weighted_indices(self.indices.shape[0], self.sample_weights)
            indices = self.indices[np.random.permutation(weighted)]
        self._indices = indices
        return self

    def __next__(self) -> tensor_dict_type:
        self._cursor += 1
        if self._cursor == len(self):
            raise StopIteration
        # indices
        start = self._cursor * self.batch_size
        end = start + self.batch_size
        # if the next batch is the last batch and its size is too small,
        #  merge it to the current batch
        if 0 < self.num_samples - end <= 0.5 * self.batch_size:
            end = self.num_samples
            self._cursor += 1
        ticks_indices, objects_indices = self._indices[start:end]
        x_ticks_indices = ticks_indices - (self.cfg.x_window - 1)
        # x & y
        x_batch = self.data.x[x_ticks_indices, ..., objects_indices, :]
        if self.cfg.y_window == 0:
            y_ticks_indices = x_ticks_indices
        else:
            y_ticks_indices = ticks_indices + self.cfg.xy_padding + 1
        labels = self.data.y[y_ticks_indices, ..., objects_indices]  # type: ignore
        # protocol
        batch_indices = ticks_indices * self.num_objects + objects_indices
        batch_indices = batch_indices.reshape([-1, 1])
        x_batch, labels = self._handle_nan(x_batch, labels)
        sample = self.callback.process_batch(x_batch, labels)
        sample[BATCH_INDICES_KEY] = to_torch(batch_indices)
        return sample

    def disable_shuffle(self) -> None:
        self.shuffle = False

    def recover_shuffle(self) -> None:
        self.shuffle = self.shuffle_backup

    def copy(self) -> "TimeSeriesLoader":
        return TimeSeriesLoader(
            self.cfg,
            self.bundle,
            is_clf=self.is_clf,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            sample_weights=self.sample_weights,
        )

    @property
    def num_samples(self) -> int:
        return len(self.indices)

    def _handle_nan(
        self,
        x_batch: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.cfg.nan_ratio <= 0.0:
            return x_batch, labels
        nan_fill = self.cfg.nan_fill
        x_nan, y_nan = np.isnan(x_batch), np.isnan(labels)
        if isinstance(nan_fill, float):
            x_batch[x_nan] = labels[y_nan] = nan_fill
        else:
            msg = f"nan fill method '{nan_fill}' is not defined"
            raise NotImplementedError(msg)
        if self.cfg.y_window == 0:
            labels = labels[..., [-1]]
        return x_batch, labels


__all__ = [
    "DataCallback",
    "FilterFunctions",
    "TimeSeriesLoader",
]
