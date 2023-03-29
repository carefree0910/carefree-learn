import csv

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from collections import Counter
from dataclasses import dataclass
from cftool.misc import print_info
from cftool.misc import print_warning
from cftool.misc import shallow_copy_dict
from cftool.misc import ISerializable
from cftool.misc import PureFromInfoMixin
from cftool.array import is_float

from ....types import data_type
from ....schema import DataTypes
from ....schema import DataBundle
from ....schema import INoInitDataBlock


TArrayPair = Tuple[np.ndarray, Optional[np.ndarray]]


class Converter(PureFromInfoMixin, ISerializable):
    column_name: str
    is_label: bool
    dtype: DataTypes
    mapping: Dict[str, int]
    counter: Dict[str, int]
    default_value: Optional[int]
    has_nan: bool
    has_empty: bool

    def to_info(self) -> Dict[str, Any]:
        return dict(
            column_name=self.column_name,
            is_label=self.is_label,
            dtype=self.dtype,
            mapping=self.mapping,
            counter=self.counter,
            default_value=self.default_value,
            has_nan=self.has_nan,
            has_empty=self.has_empty,
        )

    def _fit_with_custom_mapping(
        self,
        line: List[str],
        column_name: str,
        custom_mapping: Dict[str, int],
    ) -> None:
        self.mapping = shallow_copy_dict(custom_mapping)
        counter: Counter = Counter()
        for e in line:
            if e.lower() == "nan":
                e = "nan"
            if e in self.mapping:
                counter[e] += 1
            else:
                if self.default_value is None:
                    raise ValueError(
                        f"[{column_name}] OOD samples detected and "
                        "`custom_mapping` is provided, please specify "
                        "the `default_value` as well"
                    )
                counter["default"] += 1
        self.counter = dict(counter)

    @classmethod
    def fit(
        cls,
        column_name: str,
        line: List[str],
        is_label: bool,
        custom_mapping: Optional[Dict[str, int]],
        default_value: Optional[int],
    ) -> "Converter":
        self: Converter = cls()
        self.column_name = column_name
        self.is_label = is_label
        not_empty = [e for e in line if e]
        self.has_empty = len(not_empty) != len(line)
        self.default_value = default_value
        try:
            array = np.array(not_empty, np.float64)
        except ValueError:
            self.has_nan = False
            self.dtype = DataTypes.STRING
            if custom_mapping is not None:
                self._fit_with_custom_mapping(line, column_name, custom_mapping)
            else:
                counter = Counter(line)
                filtered_counter: Counter = Counter()
                for e, count in counter.most_common():
                    if e.lower() == "nan":
                        e = "nan"
                        self.has_nan = True
                    filtered_counter[e] += count
                most_common = filtered_counter.most_common()
                # smaller index -> more frequent
                self.mapping = {k: i for i, (k, _) in enumerate(most_common)}
                self.counter = dict(filtered_counter)
            return self
        nan_mask = np.isnan(array)
        not_nan_mask = ~nan_mask
        not_nan = array[not_nan_mask]
        self.has_nan = len(array) != len(not_nan)
        int_array = not_nan.astype(int)
        int_float_array = int_array.astype(np.float64)
        if not np.allclose(not_nan, int_float_array):
            self.mapping = {}
            self.counter = {}
            self.dtype = DataTypes.FLOAT
        else:
            self.dtype = DataTypes.INT
            if custom_mapping is not None:
                self._fit_with_custom_mapping(line, column_name, custom_mapping)
            else:
                if self.has_nan:
                    int_array = int_array[not_nan_mask]
                unique, counts = np.unique(int_array, return_counts=True)
                self.mapping = {}
                self.counter = {}
                unique_list = unique.tolist()
                counts_list = counts.tolist()
                for i, idx in enumerate(np.argsort(counts).tolist()[::-1]):
                    str_original = str(unique_list[idx])
                    self.mapping[str_original] = i
                    self.counter[str_original] = counts_list[idx]
        return self

    @classmethod
    def get_placeholder(cls, column_name: str, is_label: bool) -> "Converter":
        self: Converter = cls()
        self.column_name = column_name
        self.is_label = is_label
        self.dtype = DataTypes.FLOAT
        self.mapping = {}
        self.counter = {}
        self.default_value = None
        self.has_nan = False
        self.has_empty = False
        return self

    def convert(self, line: List[str], *, verbose: bool) -> np.ndarray:
        if self.dtype == DataTypes.FLOAT:
            return np.array([e if e else "nan" for e in line], np.float64)
        ood = 0
        method = None
        converted = []
        for e in line:
            if e.lower() == "nan":
                e = "nan"
            elif self.dtype == DataTypes.INT:
                e = str(int(e))
            mapped = self.mapping.get(e)
            if mapped is None:
                ood += 1
                if self.default_value is not None:
                    method = f"replaced with default value ({self.default_value})"
                    mapped = self.default_value
                elif self.has_nan:
                    method = "replaced with nan"
                    mapped = self.mapping["nan"]
                elif self.has_empty:
                    method = "replaced with empty string"
                    mapped = self.mapping[""]
                else:
                    method = "replaced with most frequent"
                    mapped = 0
            converted.append(mapped)
        if ood > 0:
            ratio = ood / len(converted)
            if verbose:
                print_warning(
                    f"[{self.column_name}] OOD samples detected "
                    f"({ood}/{len(converted)}={ratio:8.6f}), {method}"
                )
        return np.array(converted, np.float64)

    def __str__(self) -> str:
        postfix = (
            "" if self.dtype == DataTypes.STRING else f", has_empty={self.has_empty}"
        )
        return f"Converter({self.dtype}{postfix})"

    __repr__ = __str__


@dataclass
class MLFileProcessorConfig:
    delimiter: str = ","
    has_header: bool = True
    label_names: Optional[List[str]] = None
    label_indices: Optional[List[int]] = None
    contain_labels: bool = True
    auto_convert_labels: bool = True
    custom_mappings: Optional[Dict[str, Dict[str, int]]] = None
    default_values: Optional[Dict[str, int]] = None


get_x_column_name = lambda idx: f"f{idx}"
get_y_column_name = lambda idx: f"l{idx}"


@INoInitDataBlock.register("ml_file_parser")
class FileParserBlock(INoInitDataBlock):
    config: MLFileProcessorConfig  # type: ignore
    delimiter: str
    has_header: bool
    contain_labels: bool
    auto_convert_labels: bool
    custom_mappings: Optional[Dict[str, Dict[str, int]]]
    default_values: Optional[Dict[str, int]] = None
    feature_header: List[str]
    label_header: List[str]
    all_header: List[str]
    converters: Dict[str, Converter]
    is_placeholder: bool

    # inheritance

    def to_info(self) -> Dict[str, Any]:
        return dict(
            delimiter=self.delimiter,
            has_header=self.has_header,
            contain_labels=self.contain_labels,
            auto_convert_labels=self.auto_convert_labels,
            custom_mappings=self.custom_mappings,
            default_values=self.default_values,
            feature_header=self.feature_header,
            label_header=self.label_header,
            all_header=self.all_header,
            converters={k: v.to_info() for k, v in self.converters.items()},
            is_placeholder=self.is_placeholder,
        )

    def from_info(self, info: Dict[str, Any]) -> None:
        super().from_info(info)
        for k, v in self.converters.items():
            converter = Converter()
            converter.from_info(v)
            self.converters[k] = converter

    def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
        return self._transform(bundle, for_inference)

    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        self.delimiter = self.config.delimiter
        self.has_header = self.config.has_header
        self.contain_labels = self.config.contain_labels
        self.auto_convert_labels = self.config.auto_convert_labels
        self.custom_mappings = self.config.custom_mappings or {}
        self.default_values = self.config.default_values or {}
        if not self._check_file(bundle.x_train):
            self.is_placeholder = True
            self._setup_none_file(bundle.x_train, bundle.y_train)
            return bundle
        self.is_placeholder = False
        # sanity check
        if bundle.y_train is not None:
            raise ValueError("`y_train` should not be provided for `FileParserBlock`")
        if bundle.y_valid is not None:
            raise ValueError("`y_valid` should not be provided for `FileParserBlock`")
        # read raw train data
        if not isinstance(bundle.x_train, str):
            raise ValueError("`FileParserBlock` should be used on file inputs")
        all_header, data = self._read(bundle.x_train)
        num_columns = len(data[0])
        # read raw valid data
        if bundle.x_valid is None:
            valid_data = None
        else:
            if not isinstance(bundle.x_valid, str):
                raise ValueError("`FileParserBlock` should be used on file inputs")
            all_valid_header, valid_data = self._read(bundle.x_valid)
            if self.config.has_header and all_header != all_valid_header:
                raise ValueError(
                    "train header does not match valid header:\n"  # type: ignore
                    f"> train header | {', '.join(all_header)}\n"  # type: ignore
                    f"> valid header | {', '.join(all_valid_header)}\n"  # type: ignore
                )
        # handle label columns & header
        if not self.config.contain_labels:
            msg = "`contain_labels` should be True in `FileParserBlock.fit_transform`"
            raise ValueError(msg)
        label_names = self.config.label_names
        label_indices = self.config.label_indices
        if not label_names and not label_indices:
            msg = "neither `label_names` nor `label_indices` is provided, `[-1]` will be used"
            print_warning(msg)
            label_indices = self.config.label_indices = [-1]
        if label_indices is not None:
            label_indices = [i if i >= 0 else num_columns + i for i in label_indices]
        if all_header is not None:
            if label_names is not None:
                header = [e for e in all_header if e not in label_names]
                label_header = label_names
            else:
                header = [e for i, e in enumerate(all_header) if i not in label_indices]  # type: ignore
                label_header = [all_header[i] for i in label_indices]  # type: ignore
        else:
            if label_names:
                raise ValueError(
                    f"header is not found at `{bundle.x_train}` but `label_names` is provided. "
                    f"please either provide header in `{bundle.x_train}`, "
                    "or use `label_indices` instead."
                )
            header = []
            label_header = []
            all_header = []
            x_counter = y_counter = 0
            for i in range(num_columns):
                if i not in label_indices:  # type: ignore
                    header.append(get_x_column_name(x_counter))
                    all_header.append(header[-1])
                    x_counter += 1
                else:
                    label_header.append(get_y_column_name(y_counter))
                    all_header.append(label_header[-1])
                    y_counter += 1
        self.feature_header = header
        self.label_header = label_header
        self.all_header = all_header
        # recognize data types
        data_T = list(zip(*data))
        self.converters = {}
        for column_name, line in zip(all_header, data_T):
            is_label = column_name in label_header
            custom_mapping = self.custom_mappings.get(column_name)
            default_value = self.default_values.get(column_name)
            args = column_name, line, is_label, custom_mapping, default_value
            converter = Converter.fit(*args)  # type: ignore
            self.converters[column_name] = converter
        # transform
        train_data_T = data_T
        valid_data_T = None if valid_data is None else list(zip(*valid_data))
        return self._transform(bundle, False, (train_data_T, valid_data_T))

    def recover_labels(self, y: np.ndarray) -> np.ndarray:
        label_converters = [self.converters[h] for h in self.label_header]
        # check bypass
        if len(label_converters) != y.shape[1]:
            return y
        if len(label_converters) == 1:
            converter = label_converters[0]
            if converter.dtype == DataTypes.FLOAT:
                return y
            if is_float(y):
                return y
        recovered = []
        for i, converter in enumerate(label_converters):
            iy = y[..., i]
            if converter.dtype == DataTypes.FLOAT:
                recovered.append(iy)
            else:
                rev_mapping: Dict[int, Union[int, str]]
                if converter.dtype == DataTypes.STRING:
                    rev_mapping = {v: k for k, v in converter.mapping.items()}
                else:
                    rev_mapping = {v: int(k) for k, v in converter.mapping.items()}
                recovered.append(np.array([rev_mapping[e] for e in iy]))
        return np.ascontiguousarray(np.vstack(recovered).T)

    # internal

    def _check_file(self, x: data_type) -> bool:
        if not isinstance(x, str):
            if self.is_local_rank_0:
                print_warning("data is not a file, `FileParserBlock` will do nothing")
            return False
        return True

    def _setup_none_file(self, x: np.ndarray, y: np.ndarray) -> None:
        self.feature_header = list(map(get_x_column_name, range(x.shape[-1])))
        self.label_header = list(map(get_y_column_name, range(y.shape[-1])))
        self.all_header = self.feature_header + self.label_header
        self.converters = {}
        for h in self.feature_header:
            self.converters[h] = Converter.get_placeholder(h, False)
        for h in self.label_header:
            self.converters[h] = Converter.get_placeholder(h, True)

    def _read(self, file: str) -> Tuple[Optional[List[str]], List[List[str]]]:
        with open(file, "r") as f:
            data = list(csv.reader(f, delimiter=self.delimiter))
            for i in range(len(data) - 1, -1, -1):
                if not data[i]:
                    data.pop(i)
        if not self.has_header:
            header = None
        else:
            header = [e.strip() for e in data.pop(0)]
        return header, data

    def _convert_data(self, data_T: List[List[str]]) -> TArrayPair:
        x_list: List[np.ndarray] = []
        y_list: Optional[List[np.ndarray]] = [] if self.contain_labels else None
        header = self.all_header if self.contain_labels else self.feature_header
        for column_name, line in zip(header, data_T):
            converter = self.converters.get(column_name)
            if converter is None:
                raise ValueError(
                    f"unrecognized column name '{column_name}' occurred, "
                    f"supported column names are: {', '.join(self.all_header)}"
                )
            array = converter.convert(line, verbose=self.is_local_rank_0).ravel()
            if converter.is_label and self.auto_convert_labels:
                if converter.dtype != DataTypes.FLOAT:
                    array = array.astype(int)
            (y_list if converter.is_label else x_list).append(array)  # type: ignore
        if y_list is None:
            return np.ascontiguousarray(np.vstack(x_list).T), None
        return tuple(  # type: ignore
            map(
                np.ascontiguousarray,
                map(np.transpose, map(np.vstack, [x_list, y_list])),
            )
        )

    def _transform(
        self,
        bundle: DataBundle,
        for_inference: bool,
        train_valid_T: Optional[Tuple[Any, Any]] = None,
    ) -> DataBundle:
        if not self._check_file(bundle.x_train):
            return bundle
        if not for_inference and not self.contain_labels:
            raise ValueError("`contain_labels` should be True when not `for_inference`")
        if train_valid_T is not None:
            train_T, valid_T = train_valid_T
        else:
            if not isinstance(bundle.x_train, str):
                raise ValueError("`FileParserBlock` should be used on file inputs")
            train_T = list(zip(*self._read(bundle.x_train)[1]))
            if bundle.x_valid is None:
                valid_T = None
            else:
                if not isinstance(bundle.x_valid, str):
                    raise ValueError("`FileParserBlock` should be used on file inputs")
                valid_T = list(zip(*self._read(bundle.x_valid)[1]))
        if len(train_T) != len(self.all_header):
            if for_inference and len(train_T) + 1 == len(self.all_header):
                if self.is_local_rank_0:
                    print_info(
                        "labels are not detected and `for_inference` is set to True, "
                        "so `contain_labels` will be set to False"
                    )
                self.contain_labels = False
            else:
                raise ValueError(
                    f"number of columns ({len(train_T)}) does not match "
                    f"current number of columns ({len(self.all_header)})"
                )
        x_train, y_train = self._convert_data(train_T)
        if valid_T is None:
            x_valid = y_valid = None
        else:
            x_valid, y_valid = self._convert_data(valid_T)
        return DataBundle(x_train, y_train, x_valid, y_valid)


__all__ = [
    "MLFileProcessorConfig",
    "FileParserBlock",
]
