import torch

import numpy as np
import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Optional
from typing import NamedTuple
from cftool.misc import shallow_copy_dict
from cftool.misc import WithRegister

from .encoders import Encoder
from .encoders import EncodingResult
from ...types import tensor_dict_type
from ...protocol import StepOutputs
from ...protocol import TrainerState
from ...protocol import MetricsOutputs
from ...protocol import DataLoaderProtocol
from ...protocol import ModelWithCustomSteps
from ...constants import INPUT_KEY
from ...constants import PREDICTIONS_KEY
from ...constants import BATCH_INDICES_KEY
from ...misc.toolkit import to_numpy
from ...misc.toolkit import LoggingMixinWithRank
from ...modules.blocks import get_clones
from ...modules.blocks import Linear
from ...modules.blocks import MixedStackedEncoder


NUMERICAL_KEY = "_numerical"
ONE_HOT_KEY = "_one_hot"
EMBEDDING_KEY = "_embedding"
MERGED_KEY = "_merged"
ml_core_dict: Dict[str, Type["MLCoreProtocol"]] = {}


class SplitFeatures(NamedTuple):
    categorical: Optional[EncodingResult]
    numerical: Optional[Tensor]

    def merge(
        self,
        use_one_hot: bool = True,
        use_embedding: bool = True,
        only_categorical: bool = False,
    ) -> Tensor:
        if use_embedding and use_one_hot:
            return self._merge_all(only_categorical)
        numerical = None if only_categorical else self.numerical
        if not use_embedding and not use_one_hot:
            if only_categorical:
                raise ValueError(
                    "`only_categorical` is set to True, "
                    "but neither `one_hot` nor `embedding` is used"
                )
            assert numerical is not None
            return numerical
        categorical = self.categorical
        if not categorical:
            if only_categorical:
                raise ValueError("categorical is not available")
            assert numerical is not None
            return numerical
        if not use_one_hot:
            embedding = categorical.embedding
            assert embedding is not None
            if numerical is None:
                return embedding
            return torch.cat([numerical, embedding], dim=1)
        one_hot = categorical.one_hot
        assert not use_embedding and one_hot is not None
        if numerical is None:
            return one_hot
        return torch.cat([numerical, one_hot], dim=1)

    def _merge_all(self, only_categorical: bool) -> Tensor:
        categorical = self.categorical
        if categorical is None:
            if only_categorical:
                raise ValueError("categorical is not available")
            assert self.numerical is not None
            return self.numerical
        merged = categorical.merged
        if only_categorical or self.numerical is None:
            return merged
        return torch.cat([self.numerical, merged], dim=1)


class Dimensions(LoggingMixinWithRank):
    def __init__(
        self,
        encoder: Optional[Encoder],
        numerical_columns_mapping: Dict[int, int],
        categorical_columns_mapping: Dict[int, int],
        num_history: int,
    ):
        self.encoder = encoder
        self._categorical_dim = 0 if encoder is None else encoder.merged_dim
        self.numerical_columns_mapping = numerical_columns_mapping
        self.categorical_columns_mapping = categorical_columns_mapping
        self._numerical_columns = sorted(numerical_columns_mapping.values())
        self.num_history = num_history

    @property
    def merged_dim(self) -> int:
        return self._categorical_dim + self.numerical_dim

    @property
    def one_hot_dim(self) -> int:
        if self.encoder is None:
            return 0
        return self.encoder.one_hot_dim

    @property
    def embedding_dim(self) -> int:
        if self.encoder is None:
            return 0
        return self.encoder.embedding_dim

    @property
    def categorical_dims(self) -> Dict[int, int]:
        dims: Dict[int, int] = {}
        if self.encoder is None:
            return dims
        return self.encoder.merged_dims

    @property
    def numerical_dim(self) -> int:
        return len(self._numerical_columns)

    @property
    def has_numerical(self) -> bool:
        return self.numerical_dim > 0

    def split_features(
        self,
        x_batch: Tensor,
        batch_indices: Optional[np.ndarray],
        loader_name: Optional[str],
    ) -> SplitFeatures:
        if self.encoder is None:
            return SplitFeatures(None, x_batch)
        encoding_result = self.encoder(x_batch, batch_indices, loader_name)
        numerical_columns = self._numerical_columns
        if not numerical_columns:
            numerical = None
        else:
            numerical = x_batch[..., numerical_columns]
        return SplitFeatures(encoding_result, numerical)


class Transform(nn.Module):
    def __init__(
        self,
        dimensions: Dimensions,
        *,
        one_hot: bool,
        embedding: bool,
        only_categorical: bool,
    ):
        super().__init__()
        self.dimensions = dimensions
        self.use_one_hot = one_hot
        self.use_embedding = embedding
        self.only_categorical = only_categorical

    @property
    def out_dim(self) -> int:
        out_dim = self.dimensions.merged_dim
        if not self.use_one_hot:
            out_dim -= self.dimensions.one_hot_dim
        if not self.use_embedding:
            out_dim -= self.dimensions.embedding_dim
        if self.only_categorical:
            out_dim -= self.dimensions.numerical_dim
        return out_dim

    def forward(self, split: SplitFeatures) -> Tensor:
        return split.merge(self.use_one_hot, self.use_embedding, self.only_categorical)

    def extra_repr(self) -> str:
        one_hot_str = f"(use_one_hot): {self.use_one_hot}"
        embedding_str = f"(use_embedding): {self.use_embedding}"
        only_str = f"(only_categorical): {self.only_categorical}"
        return f"{one_hot_str}\n{embedding_str}\n{only_str}"


class MLCoreProtocol(nn.Module, WithRegister["MLCoreProtocol"], metaclass=ABCMeta):
    d = ml_core_dict

    custom_train_step: bool = False
    custom_evaluate_step: bool = False

    def __init__(self, in_dim: int, out_dim: int, num_history: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_history = num_history

    def _init_with_trainer(self, trainer: Any) -> None:
        pass

    @abstractmethod
    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        pass

    def train_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: Any,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        pass

    def evaluate_step(
        self,
        loader: DataLoaderProtocol,
        portion: float,
        trainer: Any,
    ) -> MetricsOutputs:
        pass


class MixedStackedModel(MLCoreProtocol):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_history: int,
        latent_dim: int,
        *,
        token_mixing_type: str,
        token_mixing_config: Optional[Dict[str, Any]] = None,
        channel_mixing_type: str = "ff",
        channel_mixing_config: Optional[Dict[str, Any]] = None,
        num_layers: int = 4,
        dropout: float = 0.0,
        norm_type: Optional[str] = "batch_norm",
        feedforward_dim_ratio: float = 1.0,
        sequence_pool: bool = False,
        use_head_token: bool = False,
        use_positional_encoding: bool = False,
    ):
        super().__init__(in_dim, out_dim, num_history)
        self.to_encoder = Linear(in_dim, latent_dim)
        self.encoder = MixedStackedEncoder(
            latent_dim,
            num_history,
            token_mixing_type=token_mixing_type,
            token_mixing_config=token_mixing_config,
            channel_mixing_type=channel_mixing_type,
            channel_mixing_config=channel_mixing_config,
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            feedforward_dim_ratio=feedforward_dim_ratio,
            sequence_pool=sequence_pool,
            use_head_token=use_head_token,
            use_positional_encoding=use_positional_encoding,
        )
        self.head = Linear(latent_dim, out_dim)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[MERGED_KEY]
        net = self.to_encoder(net)
        net = self.encoder(net)
        net = self.head(net)
        return {PREDICTIONS_KEY: net}


class MLModel(ModelWithCustomSteps):
    core: Union[MLCoreProtocol, nn.ModuleList]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_history: int,
        *,
        encoder: Optional[Encoder],
        numerical_columns_mapping: Dict[int, int],
        categorical_columns_mapping: Dict[int, int],
        use_one_hot: bool,
        use_embedding: bool,
        only_categorical: bool,
        core_name: str,
        core_config: Dict[str, Any],
        pre_process_batch: bool = True,
        num_repeat: Optional[int] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.encoder = encoder
        self.dimensions = Dimensions(
            self.encoder,
            numerical_columns_mapping,
            categorical_columns_mapping,
            num_history,
        )
        self.transform = Transform(
            self.dimensions,
            one_hot=use_one_hot,
            embedding=use_embedding,
            only_categorical=only_categorical,
        )
        core_config["in_dim"] = self.transform.out_dim
        core_config["out_dim"] = out_dim
        core_config["num_history"] = num_history
        core = ml_core_dict[core_name](**core_config)
        self._pre_process_batch = pre_process_batch
        if num_repeat is None:
            self.core = core
        else:
            self.core = get_clones(core, num_repeat)  # type: ignore
        self.__identifier__ = core_name
        self._num_repeat = num_repeat
        # custom steps
        self.custom_train_step = core.custom_train_step
        self.custom_evaluate_step = core.custom_evaluate_step

    def _init_with_trainer(self, trainer: Any) -> None:
        if isinstance(self.core, nn.ModuleList):
            raise ValueError("`num_repeat` is not supported for custom models")
        self.core._init_with_trainer(trainer)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        if self._pre_process_batch:
            batch_indices = batch.get(BATCH_INDICES_KEY)
            if batch_indices is not None:
                batch_indices = to_numpy(batch_indices)
            split = self.dimensions.split_features(
                batch[INPUT_KEY],
                batch_indices,
                kwargs.get("loader_name"),
            )
            batch[NUMERICAL_KEY] = split.numerical
            if split.categorical is None:
                batch[ONE_HOT_KEY] = None
                batch[EMBEDDING_KEY] = None
            else:
                batch[ONE_HOT_KEY] = split.categorical.one_hot
                batch[EMBEDDING_KEY] = split.categorical.embedding
            batch[MERGED_KEY] = self.transform(split)
        if self._num_repeat is None:
            return self.core(batch_idx, batch, state, **kwargs)
        all_results: Dict[str, List[torch.Tensor]] = {}
        for m in self.core:  # type: ignore
            m_batch = shallow_copy_dict(batch)
            m_kwargs = shallow_copy_dict(kwargs)
            sub_results = m(batch_idx, m_batch, state, **m_kwargs)
            for k, v in sub_results.items():
                all_results.setdefault(k, []).append(v)
        final_results: tensor_dict_type = {}
        for k in sorted(all_results):
            final_results[k] = torch.stack(all_results[k]).mean(0)
        return final_results

    def train_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: Any,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        if isinstance(self.core, nn.ModuleList):
            raise ValueError("`num_repeat` is not supported for custom models")
        return self.core.train_step(
            batch_idx,
            batch,
            trainer,
            forward_kwargs,
            loss_kwargs,
        )

    def evaluate_step(
        self,
        loader: DataLoaderProtocol,
        portion: float,
        trainer: Any,
    ) -> MetricsOutputs:
        if isinstance(self.core, nn.ModuleList):
            raise ValueError("`num_repeat` is not supported for custom models")
        return self.core.evaluate_step(loader, portion, trainer)


__all__ = [
    "MERGED_KEY",
    "ONE_HOT_KEY",
    "EMBEDDING_KEY",
    "NUMERICAL_KEY",
    "SplitFeatures",
    "Dimensions",
    "Transform",
    "MLCoreProtocol",
    "MixedStackedModel",
    "MLModel",
]
