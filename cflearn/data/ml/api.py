import numpy as np

from abc import ABCMeta
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Optional
from dataclasses import dataclass
from cftool.misc import safe_execute
from cftool.types import np_dict_type

from ..blocks import MLBatch
from ..blocks import MLDatasetTag
from ..blocks import GatherBlock
from ..blocks import FileParserBlock
from ..blocks import MLFileProcessorConfig
from ..blocks import MLSplitterConfig
from ..blocks import SplitterBlock
from ..blocks import RecognizerBlock
from ..blocks import MLRecognizerConfig
from ..blocks import NanHandlerBlock
from ..blocks import MLNanHandlerConfig
from ..blocks import PreProcessorBlock
from ..blocks import MLPreProcessConfig
from ..utils import ArrayLoader
from ..utils import IArrayDataset
from ...types import data_type
from ...types import sample_weights_type
from ...schema import IData
from ...schema import IDataBlock
from ...schema import DataConfig
from ...schema import DataProcessor
from ...schema import DataProcessorConfig
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY


@dataclass
@DataProcessorConfig.register("ml.bundled")
class MLBundledProcessorConfig(
    MLSplitterConfig,
    MLPreProcessConfig,
    MLRecognizerConfig,
    MLNanHandlerConfig,
    MLFileProcessorConfig,
    DataProcessorConfig,
):
    """This config is designed to bundle all possible capabilities (e.g. carefree)."""

    @property
    def default_blocks(self) -> List[IDataBlock]:
        return [
            FileParserBlock(),
            NanHandlerBlock(),
            RecognizerBlock(),
            PreProcessorBlock(),
            SplitterBlock(),
        ]


@dataclass
@DataProcessorConfig.register("ml.advanced")
class MLAdvancedProcessorConfig(MLBundledProcessorConfig):
    """
    This config is designed to be capable of any situation,
    but requires users have deeper understanding of the Processor system.
    > e.g. they often need to sepcify the `block_names` explicitly.
    """

    @property
    def default_blocks(self) -> List[IDataBlock]:
        return []


@DataProcessor.register("ml")
class MLDataProcessor(DataProcessor):
    def before_build_in_init(self) -> None:
        self.config.add_blocks(GatherBlock())

    def get_num_samples(self, x: np.ndarray, tag: MLDatasetTag) -> int:
        return len(x)

    def fetch_batch(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray],
        indices: Union[int, List[int], np.ndarray],
        tag: MLDatasetTag,
    ) -> MLBatch:
        return MLBatch(x[indices], None if y is None else y[indices])


class MLDataset(IArrayDataset):
    others: np_dict_type

    def __init__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray],
        tag: MLDatasetTag,
        processor: MLDataProcessor,
        **others: np.ndarray,
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.tag = tag
        self.processor = processor
        self.others = others

    def __len__(self) -> int:
        return self.processor.get_num_samples(self.x, self.tag)

    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> np_dict_type:
        kw = dict(x=self.x, y=self.y, indices=item, tag=self.tag)
        ml_batch = safe_execute(self.processor.fetch_batch, kw)
        batch = {
            INPUT_KEY: ml_batch.input,
            LABEL_KEY: ml_batch.labels,
        }
        if ml_batch.others is not None:
            for k, v in ml_batch.others.items():
                batch[k] = v
        for k, v in self.others.items():
            batch[k] = v[item]
        batch = self.processor.postprocess_item(batch)
        return batch


class MLLoader(ArrayLoader):
    pass


@dataclass
@DataConfig.register("ml")
class MLDataConfig(DataConfig):
    batch_size: int = 128
    valid_batch_size: int = 256


@IData.register("ml")
class MLData(IData["MLData"], metaclass=ABCMeta):
    processor: MLDataProcessor
    train_dataset: MLDataset
    valid_dataset: Optional[MLDataset]

    # inheritance

    @property
    def config_base(self) -> Type[MLDataConfig]:
        return MLDataConfig

    @property
    def processor_base(self) -> Type[MLDataProcessor]:
        return MLDataProcessor

    def get_loaders(self) -> Tuple[MLLoader, Optional[MLLoader]]:
        if not self.processor.is_ready:
            raise ValueError(
                "`processor` should be ready before calling `initialize`, "
                "did you forget to call the `prepare` method first?"
            )
        if self.bundle is None:
            raise ValueError(
                "`bundle` property is not initialized, "
                "did you forget to call the `fit` method first?"
            )
        self.train_dataset = MLDataset(
            self.bundle.x_train,
            self.bundle.y_train,
            MLDatasetTag.TRAIN,
            self.processor,
            **(self.bundle.train_others or {}),
        )
        if self.bundle.x_valid is None and self.bundle.y_valid is None:
            self.valid_dataset = None
        else:
            self.valid_dataset = MLDataset(
                self.bundle.x_valid,
                self.bundle.y_valid,
                MLDatasetTag.VALID,
                self.processor,
                **(self.bundle.valid_others or {}),
            )
        train_loader = MLLoader(
            self.train_dataset,
            shuffle=self.config.shuffle_train,
            batch_size=self.config.batch_size,
            sample_weights=self.train_weights,
        )
        if self.valid_dataset is None:
            valid_loader = None
        else:
            # when `for_inference` is True, `valid_data` will always be `None`
            # so we don't need to condition `name` field here
            valid_loader = MLLoader(
                self.valid_dataset,
                shuffle=self.config.shuffle_valid,
                batch_size=self.config.valid_batch_size or self.config.batch_size,
                sample_weights=self.valid_weights,
            )
        return train_loader, valid_loader

    # api

    @property
    def num_features(self) -> int:
        if not self.processor.is_ready:
            raise ValueError("`processor` is not ready yet")
        return self.processor.get_block(GatherBlock).num_features

    @property
    def num_labels(self) -> int:
        if not self.processor.is_ready:
            raise ValueError("`processor` is not ready yet")
        return self.processor.get_block(GatherBlock).num_labels

    def build_loader(
        self,
        x: data_type,
        y: Optional[data_type] = None,
        *,
        shuffle: bool = False,
        batch_size: Optional[int] = None,
        sample_weights: sample_weights_type = None,
        **others: np.ndarray,
    ) -> MLLoader:
        self = self.copy()
        bundle = self.transform(x, y)
        x, y = bundle.x_train, bundle.y_train
        dataset = MLDataset(x, y, MLDatasetTag.TRAIN, self.processor, **others)
        loader = MLLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size
            or self.config.valid_batch_size
            or self.config.batch_size,
            sample_weights=sample_weights,
        )
        return loader


__all__ = [
    "MLBundledProcessorConfig",
    "MLAdvancedProcessorConfig",
    "MLDataProcessor",
    "MLDataset",
    "MLLoader",
    "MLDataConfig",
    "MLData",
]
