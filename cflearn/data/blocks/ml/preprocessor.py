import numpy as np

from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from dataclasses import dataclass
from cftool.misc import safe_execute
from cftool.misc import shallow_copy_dict
from cftool.array import normalize
from cftool.array import normalize_from
from cftool.array import recover_normalize_from
from cftool.array import min_max_normalize
from cftool.array import min_max_normalize_from
from cftool.array import recover_min_max_normalize_from
from cftool.array import quantile_normalize
from cftool.array import quantile_normalize_from
from cftool.array import recover_quantile_normalize_from

from .recognizer import RecognizerBlock
from ....schema import DataBundle
from ....schema import ColumnTypes
from ....schema import INoInitDataBlock


class PreProcessMethod(str, Enum):
    MIN_MAX = "min_max"
    NORMALIZE = "normalize"
    QUANTILE_NORMALIZE = "quantile_normalize"


fn_mapping = {
    PreProcessMethod.MIN_MAX: min_max_normalize,
    PreProcessMethod.NORMALIZE: normalize,
    PreProcessMethod.QUANTILE_NORMALIZE: quantile_normalize,
}
fn_with_stats_mapping = {
    PreProcessMethod.MIN_MAX: min_max_normalize_from,
    PreProcessMethod.NORMALIZE: normalize_from,
    PreProcessMethod.QUANTILE_NORMALIZE: quantile_normalize_from,
}
recover_fn_mapping = {
    PreProcessMethod.MIN_MAX: recover_min_max_normalize_from,
    PreProcessMethod.NORMALIZE: recover_normalize_from,
    PreProcessMethod.QUANTILE_NORMALIZE: recover_quantile_normalize_from,
}


@dataclass
class MLPreProcessConfig:
    auto_preprocess: bool = True
    preprocess_methods: Optional[Dict[str, PreProcessMethod]] = None
    preprocess_configs: Optional[Dict[str, Dict[str, Any]]] = None
    label_preprocess_methods: Optional[Dict[str, PreProcessMethod]] = None
    label_preprocess_configs: Optional[Dict[str, Dict[str, Any]]] = None


def _fit_transform(
    data: np.ndarray,
    target: List[int],
    auto_preprocess: bool,
    preprocess_methods: Dict[str, PreProcessMethod],
    preprocess_configs: Dict[str, Dict[str, Any]],
    methods: Dict[str, PreProcessMethod],
    all_stats: Dict[str, Dict[str, float]],
) -> None:
    for idx in target:
        str_idx = str(idx)
        method = preprocess_methods.get(str_idx)
        if method is None and not auto_preprocess:
            continue
        if method is None:
            method = PreProcessMethod.NORMALIZE
        fn = fn_mapping.get(method)
        if fn is None:
            raise ValueError(f"unrecognized method '{method}' occurred")
        kw = shallow_copy_dict(preprocess_configs.get(str_idx, {}))
        kw["arr"] = data[..., idx]
        kw["return_stats"] = True
        array, stats = safe_execute(fn, kw)
        data[..., idx] = array
        methods[str_idx] = method
        all_stats[str_idx] = stats


def _transform(
    data: np.ndarray,
    methods: Dict[str, PreProcessMethod],
    all_stats: Dict[str, Dict[str, float]],
    mapping: Dict[PreProcessMethod, Any],
) -> None:
    for str_idx, stats in all_stats.items():
        idx = int(str_idx)
        fn = mapping[methods[str_idx]]
        data[..., idx] = fn(data[..., idx], stats)


@INoInitDataBlock.register("ml_preprocessor")
class PreProcessorBlock(INoInitDataBlock):
    config: MLPreProcessConfig  # type: ignore
    methods: Dict[str, PreProcessMethod]
    stats: Dict[str, Dict[str, float]]
    label_methods: Dict[str, PreProcessMethod]
    label_stats: Dict[str, Dict[str, float]]

    # inheritance

    def to_info(self) -> Dict[str, Any]:
        return dict(
            methods=self.methods,
            stats=self.stats,
            label_methods=self.label_methods,
            label_stats=self.label_stats,
        )

    def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
        self._transform_x(bundle.x_train)
        if bundle.y_train is not None:
            self._transform_y(bundle.y_train)
        if bundle.x_valid is not None:
            self._transform_x(bundle.x_valid)
        if bundle.y_valid is not None:
            self._transform_y(bundle.y_valid)
        return bundle

    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        x_train = bundle.x_train
        y_train = bundle.y_train
        if not isinstance(x_train, np.ndarray):
            raise ValueError("`PreProcessorBlock` should be used on numpy features.")
        if not isinstance(y_train, np.ndarray):
            raise ValueError("`PreProcessorBlock` should be used on numpy labels.")
        recognizer = self.recognizer
        if recognizer is None:
            x_target = list(range(x_train.shape[-1]))
            y_target = list(range(y_train.shape[-1]))
        else:
            mapping = recognizer.index_mapping
            x_target = [
                mapping[str_idx]
                for str_idx, column_type in recognizer.feature_types.items()
                if column_type == ColumnTypes.NUMERICAL
            ]
            y_target = [
                int(str_idx)
                for str_idx, column_type in recognizer.label_types.items()
                if column_type == ColumnTypes.NUMERICAL
            ]
        if self.config.preprocess_methods is None:
            self.config.preprocess_methods = {}
        if self.config.preprocess_configs is None:
            self.config.preprocess_configs = {}
        if self.config.label_preprocess_methods is None:
            self.config.label_preprocess_methods = {}
        if self.config.label_preprocess_configs is None:
            self.config.label_preprocess_configs = {}
        self.methods = {}
        self.stats = {}
        self.label_methods = {}
        self.label_stats = {}
        _fit_transform(
            x_train,
            x_target,
            self.config.auto_preprocess,
            self.config.preprocess_methods,
            self.config.preprocess_configs,
            self.methods,
            self.stats,
        )
        _fit_transform(
            y_train,
            y_target,
            self.config.auto_preprocess,
            self.config.label_preprocess_methods,
            self.config.label_preprocess_configs,
            self.label_methods,
            self.label_stats,
        )
        if bundle.x_valid is not None:
            self._transform_x(bundle.x_valid)
        if bundle.y_valid is not None:
            self._transform_y(bundle.y_valid)
        return bundle

    def recover_labels(self, y: np.ndarray) -> np.ndarray:
        _transform(y, self.label_methods, self.label_stats, recover_fn_mapping)
        return y

    # api

    @property
    def recognizer(self) -> Optional[RecognizerBlock]:
        return self.try_get_previous(RecognizerBlock.__identifier__)

    # internal

    def _transform_x(self, x: np.ndarray) -> None:
        _transform(x, self.methods, self.stats, fn_with_stats_mapping)

    def _transform_y(self, y: np.ndarray) -> None:
        _transform(y, self.label_methods, self.label_stats, fn_with_stats_mapping)


__all__ = [
    "PreProcessMethod",
    "MLPreProcessConfig",
    "PreProcessorBlock",
]
