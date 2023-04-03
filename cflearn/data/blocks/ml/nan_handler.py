import numpy as np

from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional
from dataclasses import dataclass

from .file import TArrayPair
from .file import FileParserBlock
from ....schema import DataBundle
from ....schema import INoInitDataBlock


class NanReplaceMethod(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"


class NanDropStrategy(str, Enum):
    NONE = "none"
    DROP_Y = "drop_y"
    DROP_ALL = "drop_all"


@dataclass
class MLNanHandlerConfig:
    drop_strategy: NanDropStrategy = NanDropStrategy.DROP_Y
    replace_method: NanReplaceMethod = NanReplaceMethod.MEDIAN


def drop_nan(
    x: np.ndarray,
    y: Optional[np.ndarray],
    x_nan_mask: np.ndarray,
    y_nan_mask: Optional[np.ndarray],
    *,
    only_drop_y: bool,
) -> TArrayPair:
    if y_nan_mask is not None:
        y_nan_mask = np.any(y_nan_mask, axis=1)
    if not only_drop_y:
        x_nan_mask = np.any(x_nan_mask, axis=1)
        if y_nan_mask is None:
            nan_mask = x_nan_mask
        else:
            nan_mask = x_nan_mask | y_nan_mask
    else:
        if y_nan_mask is None:
            return x, y
        nan_mask = y_nan_mask
    valid_mask = ~nan_mask
    return x[valid_mask], None if y is None else y[valid_mask]


def replace_nan(
    data: np.ndarray,
    nan_mask: np.ndarray,
    idx: int,
    column_name: str,
    replacement: Optional[float],
) -> None:
    array = data[..., idx]
    nan_mask = nan_mask[..., idx]
    if not np.any(nan_mask):
        return
    if replacement is None:
        msg = f"nan values occurred at column {column_name} but replacement is not available"
        raise ValueError(msg)
    array[nan_mask] = replacement
    data[..., idx] = array


@INoInitDataBlock.register("ml_nan_handler")
class NanHandlerBlock(INoInitDataBlock):
    config: MLNanHandlerConfig  # type: ignore
    drop_strategy: NanDropStrategy
    has_nan: Dict[str, bool]
    replacements: Dict[str, float]
    label_has_nan: Dict[str, bool]
    label_replacements: Dict[str, float]

    # inheritance

    def to_info(self) -> Dict[str, Any]:
        return dict(
            drop_strategy=self.drop_strategy,
            has_nan=self.has_nan,
            replacements=self.replacements,
            label_has_nan=self.label_has_nan,
            label_replacements=self.label_replacements,
        )

    def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
        return self._transform(bundle, for_inference)

    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        x_tr = bundle.x_train
        y_tr = bundle.y_train
        if not isinstance(x_tr, np.ndarray):
            raise ValueError("`NanHandlerBlock` should be used on numpy features.")
        if not isinstance(y_tr, np.ndarray):
            raise ValueError("`NanHandlerBlock` should be used on numpy labels.")
        x_tr_nan_mask = np.isnan(x_tr)
        ytr_nan_mask = np.isnan(y_tr)
        self.drop_strategy = self.config.drop_strategy
        self.has_nan = {}
        self.replacements = {}
        self.label_has_nan = {}
        self.label_replacements = {}
        self._fit(self.has_nan, self.replacements, x_tr, x_tr_nan_mask)
        self._fit(self.label_has_nan, self.label_replacements, y_tr, ytr_nan_mask)
        return self._transform(
            bundle,
            False,
            x_tr_nan_mask,
            ytr_nan_mask,
            None if bundle.x_valid is None else np.isnan(bundle.x_valid),
            None if bundle.y_valid is None else np.isnan(bundle.y_valid),
        )

    # api

    @property
    def file_parser(self) -> Optional[FileParserBlock]:
        return self.try_get_previous(FileParserBlock)

    # internal

    def _fit(
        self,
        has_nan: Dict[str, bool],
        replacements: Dict[str, float],
        data: np.ndarray,
        data_nan_mask: np.ndarray,
    ) -> None:
        for idx in range(data.shape[-1]):
            str_idx = str(idx)
            array = data[..., idx]
            not_nan = array[~data_nan_mask[..., idx]]
            has_nan[str_idx] = len(array) != len(not_nan)
            if len(not_nan) == 0:
                replacements[str_idx] = 0.0
            else:
                replacements[str_idx] = getattr(np, self.config.replace_method)(not_nan)

    def _get_column_name(self, idx: int, is_label: bool) -> str:
        file_parser = self.file_parser
        if file_parser is None:
            return str(idx)
        if is_label:
            return f"'{file_parser.label_header[idx]}'"
        return f"'{file_parser.feature_header[idx]}'"

    def _transform(
        self,
        bundle: DataBundle,
        for_inference: bool,
        x_train_nan_mask: Optional[np.ndarray] = None,
        y_train_nan_mask: Optional[np.ndarray] = None,
        x_valid_nan_mask: Optional[np.ndarray] = None,
        y_valid_nan_mask: Optional[np.ndarray] = None,
    ) -> DataBundle:
        if not isinstance(bundle.x_train, np.ndarray):
            raise ValueError("`NanHandlerBlock` should be used on numpy features.")
        if bundle.y_train is not None and not isinstance(bundle.y_train, np.ndarray):
            raise ValueError("`NanHandlerBlock` should be used on numpy labels.")
        if x_train_nan_mask is None:
            x_train_nan_mask = np.isnan(bundle.x_train)
        if y_train_nan_mask is None and bundle.y_train is not None:
            y_train_nan_mask = np.isnan(bundle.y_train)
        if x_valid_nan_mask is None and bundle.x_valid is not None:
            x_valid_nan_mask = np.isnan(bundle.x_valid)
        if y_valid_nan_mask is None and bundle.y_valid is not None:
            y_valid_nan_mask = np.isnan(bundle.y_valid)
        # drop nan
        if not for_inference and self.drop_strategy != NanDropStrategy.NONE:
            only_drop_y = self.drop_strategy == NanDropStrategy.DROP_Y
            bundle.x_train, bundle.y_train = drop_nan(
                bundle.x_train,
                bundle.y_train,
                x_train_nan_mask,
                y_train_nan_mask,
                only_drop_y=only_drop_y,
            )
            if bundle.x_valid is not None:
                bundle.x_valid, bundle.y_valid = drop_nan(
                    bundle.x_valid,
                    bundle.y_valid,
                    x_valid_nan_mask,
                    y_valid_nan_mask,
                    only_drop_y=only_drop_y,
                )
        # replace nan
        for idx in range(bundle.x_train.shape[-1]):
            cn = self._get_column_name(idx, False)
            replacement = self.replacements.get(str(idx))
            replace_nan(bundle.x_train, x_train_nan_mask, idx, cn, replacement)
            if bundle.x_valid is not None:
                replace_nan(bundle.x_valid, x_valid_nan_mask, idx, cn, replacement)
        if bundle.y_train is not None:
            for idx in range(bundle.y_train.shape[-1]):
                cn = self._get_column_name(idx, True)
                replacement = self.label_replacements.get(str(idx))
                replace_nan(bundle.y_train, y_train_nan_mask, idx, cn, replacement)
                if bundle.y_valid is not None:
                    replace_nan(bundle.y_valid, y_valid_nan_mask, idx, cn, replacement)
        return bundle


__all__ = [
    "NanReplaceMethod",
    "NanDropStrategy",
    "MLNanHandlerConfig",
    "NanHandlerBlock",
]
