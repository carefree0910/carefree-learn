import numpy as np

from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from dataclasses import dataclass
from cftool.misc import print_warning

from .file import FileParserBlock
from ....schema import DataTypes
from ....schema import DataBundle
from ....schema import ColumnTypes
from ....schema import INoInitDataBlock


TRes = Tuple[ColumnTypes, int]


@dataclass
class MLRecognizerConfig:
    all_close_threshold: float = 1.0e-6
    redundancy_threshold: float = 0.5
    custom_feature_types: Optional[Dict[str, ColumnTypes]] = None


@INoInitDataBlock.register("ml_recognizer")
class RecognizerBlock(INoInitDataBlock):
    config: MLRecognizerConfig  # type: ignore
    feature_types: Dict[str, ColumnTypes]
    label_types: Dict[str, ColumnTypes]
    num_unique_features: Dict[str, int]
    num_unique_labels: Dict[str, int]
    # map from original indices to new indices (which exclude the REDUNDANT columns)
    index_mapping: Dict[str, int]

    # inheritance

    def to_info(self) -> Dict[str, Any]:
        return dict(
            feature_types=self.feature_types,
            label_types=self.label_types,
            num_unique_features=self.num_unique_features,
            num_unique_labels=self.num_unique_labels,
            index_mapping=self.index_mapping,
        )

    def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
        if not isinstance(bundle.x_train, np.ndarray):
            raise ValueError("`RecognizerBlock` should be used on numpy features")
        target = [
            int(str_idx)
            for str_idx, column_type in self.feature_types.items()
            if column_type != ColumnTypes.REDUNDANT
        ]
        bundle.x_train = bundle.x_train[..., target]
        if bundle.x_valid is not None:
            if not isinstance(bundle.x_valid, np.ndarray):
                raise ValueError("`RecognizerBlock` should be used on numpy features")
            bundle.x_valid = bundle.x_valid[..., target]
        return bundle

    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        x_train: np.ndarray = bundle.x_train
        y_train: np.ndarray = bundle.y_train
        self.feature_types = {}
        self.label_types = {}
        self.num_unique_features = {}
        self.num_unique_labels = {}
        for idx, line in enumerate(x_train.T):
            self._fit(idx, line, False, self.config.custom_feature_types)
        for idx, line in enumerate(y_train.T):
            self._fit(idx, line, True, None)
        counter = 0
        self.index_mapping = {}
        for i in range(x_train.shape[-1]):
            str_i = str(i)
            if self.feature_types[str_i] == ColumnTypes.REDUNDANT:
                continue
            self.index_mapping[str_i] = counter
            counter += 1
        return self.transform(bundle, False)

    # api

    @property
    def file_parser(self) -> Optional[FileParserBlock]:
        return self.try_get_previous(FileParserBlock)

    # internal

    def _fit(
        self,
        idx: int,
        line: np.ndarray,
        is_label: bool,
        custom_types: Optional[Dict[str, ColumnTypes]],
    ) -> None:
        str_idx = str(idx)
        file_parser = self.file_parser
        if file_parser is None or file_parser.is_placeholder:
            int_line = line.astype(int).astype(np.float64)
            dtype = DataTypes.INT if np.allclose(line, int_line) else DataTypes.FLOAT
            key = name = str_idx
        else:
            attr = f"{'label' if is_label else 'feature'}_header"
            header = getattr(file_parser, attr)
            dtype = file_parser.converters[header[idx]].dtype
            key = header[idx]
            name = f"'{key}'"
        custom_type = None if custom_types is None else custom_types.get(key)
        column_type, num_unique = self._fit_with(name, line, dtype, custom_type)
        (self.label_types if is_label else self.feature_types)[str_idx] = column_type
        if column_type != ColumnTypes.NUMERICAL:
            uniques = self.num_unique_labels if is_label else self.num_unique_features
            uniques[str_idx] = num_unique

    def _fit_with(
        self,
        name: str,
        line: np.ndarray,
        dtype: DataTypes,
        custom_type: Optional[ColumnTypes],
    ) -> TRes:
        prefix = f"values in {dtype} column {name}"
        postfix = "It'll be marked as redundant."
        if dtype == DataTypes.FLOAT:
            threshold = self.config.all_close_threshold
            if line.max().item() - line.min().item() <= threshold:
                if self.is_local_rank_0:
                    print_warning(f"{prefix} are ALL CLOSE. {postfix}")
                return ColumnTypes.REDUNDANT, 0
            if np.all(line == 0.0):
                if self.is_local_rank_0:
                    print_warning(f"{prefix} are ALL ZERO. {postfix}")
                return ColumnTypes.REDUNDANT, 0
            if custom_type is None:
                return ColumnTypes.NUMERICAL, 0
            return custom_type, 0
        num_unique = len(np.unique(line))
        num_samples = len(line)
        if num_unique == 1:
            if self.is_local_rank_0:
                print_warning(f"{prefix} are ALL SAME. {postfix}")
            return ColumnTypes.REDUNDANT, num_unique
        if num_unique == num_samples:
            if self.is_local_rank_0:
                print_warning(f"{prefix} are ALL DIFFERENT. {postfix}")
            return ColumnTypes.REDUNDANT, num_unique
        unique_ratio = num_unique / num_samples
        if unique_ratio >= self.config.redundancy_threshold:
            if self.is_local_rank_0:
                msg = f"{prefix} are TOO MANY (ratio={unique_ratio:8.6f}). {postfix}"
                print_warning(msg)
            return ColumnTypes.REDUNDANT, num_unique
        if custom_type is None:
            return ColumnTypes.CATEGORICAL, num_unique
        return custom_type, num_unique


__all__ = [
    "ColumnTypes",
    "MLRecognizerConfig",
    "RecognizerBlock",
]
