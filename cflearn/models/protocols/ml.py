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
from typing import Tuple
from typing import Union
from typing import Optional
from typing import NamedTuple
from cftool.misc import shallow_copy_dict
from cftool.misc import WithRegister
from cftool.array import to_numpy
from cftool.types import tensor_dict_type

from ...protocol import ITrainer
from ...protocol import StepOutputs
from ...protocol import TrainerState
from ...protocol import MetricsOutputs
from ...protocol import IDataLoader
from ...protocol import ModelWithCustomSteps
from ...constants import INPUT_KEY
from ...constants import BATCH_INDICES_KEY
from ...modules.blocks import get_clones
from ...modules.blocks import Linear
from ...modules.blocks import MixedStackedEncoder


NUMERICAL_KEY = "_numerical"
ONE_HOT_KEY = "_one_hot"
EMBEDDING_KEY = "_embedding"
MERGED_KEY = "_merged"
ml_core_dict: Dict[str, Type["IMLCore"]] = {}


class IEncoder:
    merged_dim: int
    merged_dims: Dict[int, int]
    one_hot_dim: int
    embedding_dim: int

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def encode(self, *args: Any, **kwds: Any) -> Any:
        pass


class EncodingResult(NamedTuple):
    one_hot: Optional[torch.Tensor]
    embedding: Optional[torch.Tensor]

    @property
    def merged(self) -> torch.Tensor:
        if self.one_hot is None and self.embedding is None:
            raise ValueError("no data is provided in `EncodingResult`")
        if self.one_hot is None:
            assert self.embedding is not None
            return self.embedding
        if self.embedding is None:
            assert self.one_hot is not None
            return self.one_hot
        return torch.cat([self.one_hot, self.embedding], dim=-1)


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
            return torch.cat([numerical, embedding], dim=-1)
        one_hot = categorical.one_hot
        assert not use_embedding and one_hot is not None
        if numerical is None:
            return one_hot
        return torch.cat([numerical, one_hot], dim=-1)

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


class IndicesResponse(NamedTuple):
    indices: Tuple[int, ...]
    is_categorical: bool


class Dimensions:
    def __init__(
        self,
        *,
        num_history: int,
        encoder: Optional[IEncoder] = None,
        numerical_columns: List[int],
        categorical_columns: List[int],
    ):
        self.encoder = encoder
        if encoder is not None:
            self.one_hot_dim = encoder.one_hot_dim
            self.embedding_dim = encoder.embedding_dim
            c_dims = self.categorical_dims = encoder.merged_dims
        else:
            self.one_hot_dim = 0
            self.embedding_dim = 0
            c_dims = self.categorical_dims = {}

        self.num_history = num_history
        self.numerical_columns = sorted(numerical_columns)
        self.categorical_columns = sorted(categorical_columns)

        self.categorical_dim = sum(c_dims[idx] for idx in categorical_columns)
        self.numerical_dim = len(numerical_columns)
        self.merged_dim = self.categorical_dim + self.numerical_dim

        self.has_categorical = self.categorical_dim > 0
        self.has_numerical = self.numerical_dim > 0

    def __str__(self) -> str:
        return "\n".join(
            [
                "Dimensions(",
                f"    merged_dim    = {self.merged_dim}",
                f"    one_hot_dim   = {self.one_hot_dim}",
                f"    embedding_dim = {self.embedding_dim}",
                f"    numerical_dim = {self.numerical_dim}",
                ")",
            ]
        )

    __repr__ = __str__

    def split_features(
        self,
        x_batch: Tensor,
        batch_indices: Optional[np.ndarray],
        loader_name: Optional[str],
    ) -> SplitFeatures:
        if self.encoder is None:
            return SplitFeatures(None, x_batch)
        encoding_result = self.encoder(x_batch, batch_indices, loader_name)
        if not self.has_numerical:
            numerical = None
        else:
            numerical = x_batch[..., self.numerical_columns]
        return SplitFeatures(encoding_result, numerical)

    def get_indices_in_merged(self, idx: int) -> Optional[IndicesResponse]:
        if idx in self.numerical_columns:
            return IndicesResponse((idx,), False)
        if idx in self.categorical_columns:
            categorical_dims = self.categorical_dims
            start_idx = self.categorical_columns.index(idx)
            start = sum(categorical_dims[i] for i in range(start_idx))
            start += self.numerical_dim
            indices = tuple(range(start, start + categorical_dims[start_idx]))
            return IndicesResponse(indices, True)
        return None


class Transform:
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

    def __call__(self, split: SplitFeatures) -> Tensor:
        return split.merge(self.use_one_hot, self.use_embedding, self.only_categorical)

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

    def extra_repr(self) -> str:
        one_hot_str = f"(use_one_hot): {self.use_one_hot}"
        embedding_str = f"(use_embedding): {self.use_embedding}"
        only_str = f"(only_categorical): {self.only_categorical}"
        return f"{one_hot_str}\n{embedding_str}\n{only_str}"


class IMLCore(nn.Module, WithRegister["IMLCore"], metaclass=ABCMeta):
    d = ml_core_dict

    custom_train_step: bool = False
    custom_evaluate_step: bool = False

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int,
        dimensions: Dimensions,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_history = num_history
        self.dimensions = dimensions

    def _init_with_trainer(self, trainer: ITrainer) -> None:
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
        trainer: ITrainer,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        pass

    def evaluate_step(
        self,
        loader: IDataLoader,
        portion: float,
        trainer: ITrainer,
    ) -> MetricsOutputs:
        pass


class MixedStackedModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
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
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_history = num_history
        self.to_encoder = Linear(input_dim, latent_dim)
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
        self.head = Linear(latent_dim, output_dim)

    def forward(self, net: Tensor) -> Tensor:
        net = self.to_encoder(net)
        net = self.encoder(net)
        net = self.head(net)
        return net


class MLModel(ModelWithCustomSteps):
    core: Union[IMLCore, nn.ModuleList]
    encoder: Union[nn.Module, nn.ModuleList]
    dimensions: Union[Dimensions, List[Dimensions]]
    transform: Union[Transform, List[Transform]]

    def __init__(
        self,
        output_dim: int,
        num_history: int,
        *,
        encoder: Optional[IEncoder],
        use_encoder_cache: bool,
        numerical_columns: List[int],
        categorical_columns: List[int],
        use_one_hot: bool,
        use_embedding: bool,
        only_categorical: bool,
        core_name: str,
        core_config: Dict[str, Any],
        pre_process_batch: bool = True,
        num_repeat: Optional[int] = None,
    ):
        def _make_dimensions(enc: Optional[IEncoder]) -> Dimensions:
            return Dimensions(
                num_history=num_history,
                encoder=enc,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
            )

        def _make_transform(dim: Dimensions) -> Transform:
            return Transform(
                dim,
                one_hot=use_one_hot,
                embedding=use_embedding,
                only_categorical=only_categorical,
            )

        super().__init__()
        self.output_dim = output_dim
        self.encoder = encoder
        self.use_encoder_cache = use_encoder_cache
        self.dimensions = _make_dimensions(encoder)
        self.transform = _make_transform(self.dimensions)
        core_config = shallow_copy_dict(core_config)
        core_config["input_dim"] = self.transform.out_dim
        core_config["output_dim"] = output_dim
        core_config["num_history"] = num_history
        core_config["dimensions"] = self.dimensions
        core = ml_core_dict[core_name](**core_config)
        self._pre_process_batch = pre_process_batch
        if num_repeat is None:
            self.core = core
        else:
            self.core = get_clones(core, num_repeat)  # type: ignore
            if encoder is None:
                self.dimensions = [self.dimensions] * num_repeat
                self.transform = [self.transform] * num_repeat
            else:
                self.encoder = get_clones(encoder, num_repeat)
                self.dimensions = list(map(_make_dimensions, self.encoder))
                self.transform = list(map(_make_transform, self.dimensions))
        self.__identifier__ = core_name
        self._num_repeat = num_repeat
        # custom steps
        self.custom_train_step = core.custom_train_step
        self.custom_evaluate_step = core.custom_evaluate_step

    def _init_with_trainer(self, trainer: ITrainer) -> None:
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
        def _get_split(
            dim: Dimensions,
            batch_indices_: Optional[np.ndarray],
        ) -> SplitFeatures:
            return dim.split_features(
                batch[INPUT_KEY],
                batch_indices_,
                kwargs.get("loader_name") if self.use_encoder_cache else None,
            )

        def _inject_batch(
            b: tensor_dict_type,
            batch_indices_: Optional[np.ndarray],
            dim: Dimensions,
            transform: Transform,
        ) -> None:
            split = _get_split(dim, batch_indices_)
            b[NUMERICAL_KEY] = split.numerical
            if split.categorical is None:
                b[ONE_HOT_KEY] = None
                b[EMBEDDING_KEY] = None
            else:
                b[ONE_HOT_KEY] = split.categorical.one_hot
                b[EMBEDDING_KEY] = split.categorical.embedding
            b[MERGED_KEY] = transform(split)

        if self._num_repeat is None:
            if self._pre_process_batch:
                batch_indices = batch.get(BATCH_INDICES_KEY)
                if batch_indices is not None:
                    batch_indices = to_numpy(batch_indices)
                _inject_batch(batch, batch_indices, self.dimensions, self.transform)  # type: ignore
            return self.core(batch_idx, batch, state, **kwargs)

        batches = []
        if not self._pre_process_batch:
            for _ in range(self._num_repeat):
                batches.append(shallow_copy_dict(batch))
        else:
            batch_indices = batch.get(BATCH_INDICES_KEY)
            if batch_indices is not None:
                batch_indices = to_numpy(batch_indices)
            for d, t in zip(self.dimensions, self.transform):  # type: ignore
                local_batch = shallow_copy_dict(batch)
                _inject_batch(local_batch, batch_indices, d, t)
                batches.append(local_batch)
        all_results: Dict[str, List[torch.Tensor]] = {}
        for m, m_batch in zip(self.core, batches):  # type: ignore
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
        trainer: ITrainer,
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
        loader: IDataLoader,
        portion: float,
        trainer: ITrainer,
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
    "IMLCore",
    "MixedStackedModel",
    "MLModel",
]
