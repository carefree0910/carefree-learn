import torch

import numpy as np

from typing import *
from abc import ABCMeta
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleDict
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from cftool.misc import register_core
from cftool.misc import context_error_handler
from cfdata.tabular import TaskTypes
from cfdata.tabular import ColumnTypes

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None

from ..modules import *
from ..types import tensor_dict_type
from ..losses import LossBase
from ..configs import Configs
from ..configs import Environment
from ..protocol import TrainerState
from ..protocol import ModelProtocol
from ..protocol import DataLoaderProtocol
from ..misc.toolkit import to_torch
from ..modules.heads import HeadBase
from ..modules.heads import HeadConfigs
from ..modules.blocks import _get_clones
from ..modules.blocks import DNDF
from ..modules.transform import transform_config_mapping
from ..modules.transform import Transform
from ..modules.transform import Dimensions
from ..modules.transform import SplitFeatures
from ..modules.extractors import ExtractorBase
from ..modules.aggregators import AggregatorBase

model_dict: Dict[str, Type["ModelBase"]] = {}


class PipeConfig(NamedTuple):
    transform: str
    extractor: str
    reuse_extractor: bool
    head: str
    extractor_config: str
    head_config: str
    extractor_meta_scope: Optional[str]
    head_meta_scope: Optional[str]

    @property
    def use_extractor_meta(self) -> bool:
        return self.extractor_meta_scope is not None

    @property
    def use_head_meta(self) -> bool:
        return self.head_meta_scope is not None

    @property
    def extractor_scope(self) -> str:
        if self.extractor_meta_scope is None:
            return self.extractor
        return self.extractor_meta_scope

    @property
    def head_scope(self) -> str:
        if self.head_meta_scope is None:
            return self.head
        return self.head_meta_scope

    @property
    def extractor_config_key(self) -> str:
        if self.extractor_meta_scope is None:
            prefix = self.extractor
        else:
            prefix = self.extractor_meta_scope
        return f"{self.transform}_{prefix}_{self.extractor_config}"

    @property
    def extractor_unique_key(self) -> str:
        prefix = f"{self.extractor_meta_scope}_{self.extractor}"
        return f"{self.transform}_{prefix}_{self.extractor_config}"

    @property
    def head_config_key(self) -> str:
        if self.head_meta_scope is None:
            prefix = self.head
        else:
            prefix = self.head_meta_scope
        return f"{prefix}_{self.head_config}"


class ModelBase(ModelProtocol, metaclass=ABCMeta):
    registered_pipes: Optional[Dict[str, PipeConfig]] = None
    registered_meta_configs: Optional[Dict[str, Dict[str, Any]]] = None

    def __init__(
        self,
        environment: Environment,
        tr_loader: DataLoaderProtocol,
        cv_loader: Optional[DataLoaderProtocol],
        tr_weights: Optional[np.ndarray],
        cv_weights: Optional[np.ndarray],
        loaded_registered_pipes: Optional[Dict[str, PipeConfig]] = None,
    ):
        super().__init__()
        # common
        self.environment = environment
        self.device = environment.device
        self.timing = environment.use_timing_context
        self.tr_loader = tr_loader
        self.cv_loader = cv_loader
        self.data = self.tr_data = tr_loader.data
        self.cv_data = None if cv_loader is None else cv_loader.data
        self.num_train = len(self.tr_data)
        self.num_valid = None if self.cv_data is None else len(self.cv_data)
        self.tr_weights, self.cv_weights = tr_weights, cv_weights
        self._preset_config()
        self._init_config()
        # loss
        self.loss_name = self.config.setdefault("loss", "auto")
        if self.loss_name == "auto":
            if self.tr_data.is_reg:
                self.loss_name = "mae"
            else:
                self.loss_name = "focal"
        loss_config = self.config.setdefault("loss_config", {})
        self.loss = LossBase.make(self.loss_name, loss_config, "none")
        # encoder
        excluded = 0
        numerical_columns_mapping = {}
        categorical_columns_mapping = {}
        categorical_dims = []
        encoding_methods = []
        encoding_configs = []
        true_categorical_columns = []
        if self.tr_data.is_simplify:
            for idx in range(self.tr_data.processed.x.shape[1]):
                numerical_columns_mapping[idx] = idx
        else:
            ts_indices = self.tr_data.ts_indices
            recognizers = self.tr_data.recognizers
            sorted_indices = [idx for idx in sorted(recognizers) if idx != -1]
            for idx in sorted_indices:
                recognizer = recognizers[idx]
                assert recognizer is not None
                if not recognizer.info.is_valid or idx in ts_indices:
                    excluded += 1
                elif recognizer.info.column_type is ColumnTypes.NUMERICAL:
                    numerical_columns_mapping[idx] = idx - excluded
                else:
                    str_idx = str(idx)
                    categorical_dims.append(recognizer.num_unique_values)
                    encoding_methods.append(
                        self.encoding_methods.setdefault(
                            str_idx, self.default_encoding_method
                        )
                    )
                    encoding_configs.append(
                        self.encoding_configs.setdefault(
                            str_idx, self.default_encoding_configs
                        )
                    )
                    true_idx = idx - excluded
                    true_categorical_columns.append(true_idx)
                    categorical_columns_mapping[idx] = true_idx
        if not true_categorical_columns:
            self.encoder = None
        else:
            loaders = {"tr": self.tr_loader}
            if self.cv_loader is not None:
                loaders["cv"] = self.cv_loader
            encoder_config = self.config.setdefault("encoder_config", {})
            self.encoder = Encoder(
                encoder_config,
                categorical_dims,
                encoding_methods,
                encoding_configs,
                true_categorical_columns,
                loaders,
            )
        # dimensions
        self.dimensions = Dimensions(
            self.encoder,
            numerical_columns_mapping,
            categorical_columns_mapping,
            self.num_history,
        )
        # pipes
        self.pipes: Dict[str, Tuple[str, str, str]] = {}
        self.transforms = ModuleDict()
        self.extractors = ModuleDict()
        self.heads = ModuleDict()
        self.bypassed_pipes: Set[str] = set()
        self._extractor_configs: Dict[str, Dict[str, Any]] = {}
        self._head_configs: Dict[str, Dict[str, Any]] = {}
        self._head_config_ins_dict: Dict[str, HeadConfigs] = {}
        if loaded_registered_pipes is not None:
            self.registered_pipes = loaded_registered_pipes
        if self.registered_pipes is None:
            raise ValueError(f"No `pipe` is registered in {type(self).__name__}")
        with torch.no_grad():
            for key, pipe_config in self.registered_pipes.items():
                local_configs = self.pipe_configs.setdefault(key, {})
                transform_config = local_configs.setdefault("transform", {})
                extractor_config = local_configs.setdefault("extractor", {})
                head_config = local_configs.setdefault("head", {})
                self.add_pipe(
                    key,
                    pipe_config,
                    transform_config,
                    extractor_config,
                    head_config,
                )
        aggregator = self.config["aggregator"]
        aggregator_config = self.config["aggregator_config"]
        self.aggregator = AggregatorBase.make(aggregator, **aggregator_config)
        # caches
        self._transform_cache: Dict[str, Tensor] = {}
        self._extractor_cache: Dict[str, Tensor] = {}

    def __getattr__(self, item: str) -> Any:
        try:
            return super().__getattr__(item)
        except AttributeError:
            value = self.config.get(item)
            if value is not None:
                return value
            value = self.environment.config.get(item)
            if value is not None:
                return value
            msg = f"attribute '{item}' is not defined in {type(self).__name__}"
            raise AttributeError(msg)

    @property
    def config(self) -> Dict[str, Any]:
        return self.environment.model_config

    @property
    def task_type(self) -> TaskTypes:
        return self.tr_data.task_type

    @property
    def labels_key(self) -> str:
        return self.tr_loader.labels_key

    @property
    def num_repeat(self) -> int:
        return self.environment.num_repeat or 1

    @property
    def num_history(self) -> int:
        num_history = 1
        if self.tr_data.is_ts:
            sampler_config = self.environment.sampler_config
            aggregation_config = sampler_config.get("aggregation_config", {})
            num_history = aggregation_config.get("num_history")
            if num_history is None:
                raise ValueError(
                    "please provide `num_history` in `aggregation_config` "
                    "in `cflearn.make` for time series tasks."
                )
        return num_history

    @property
    def pipe_configs(self) -> Dict[str, Any]:
        return self.config.setdefault("pipe_configs", {})

    def get_pipe_config(self, pipe: str, part: str) -> Dict[str, Any]:
        return self.pipe_configs.setdefault(pipe, {}).setdefault(part, {})

    def add_pipe(
        self,
        key: str,
        pipe_config: PipeConfig,
        transform_config: Dict[str, bool],
        extractor_config: Dict[str, Any],
        head_config: Dict[str, Any],
    ) -> None:
        # meta config
        meta_configs = shallow_copy_dict(self.registered_meta_configs or {})
        meta_config = shallow_copy_dict(meta_configs.get(key, {}))
        meta_transform_config = shallow_copy_dict(meta_config.get("transform", {}))
        meta_extractor_config = shallow_copy_dict(meta_config.get("extractor", {}))
        meta_head_config = shallow_copy_dict(meta_config.get("head", {}))
        # update user increment config
        if meta_config:
            user_increment_config = self.environment.user_increment_config
            user_model_config = user_increment_config.setdefault("model_config", {})
            user_pipe_configs = user_model_config.setdefault("pipe_configs", {})
            user_pipe_config = user_pipe_configs.setdefault(key, {})
            keys = ["transform", "extractor", "head"]
            metas = [meta_transform_config, meta_extractor_config, meta_head_config]
            for k, meta in zip(keys, metas):
                update_dict(meta, user_pipe_config.setdefault(k, {}))
        # transform
        transform_key = pipe_config.transform
        if transform_key in self.transforms:
            transform_exists = True
            transforms = self.transforms[transform_key]
        else:
            transform_exists = False
            transform_cfg = Configs.get("transform", transform_key, **transform_config)
            transform_kwargs = transform_cfg.pop()
            update_dict(meta_transform_config, transform_kwargs)
            transforms = _get_clones(
                Transform(self.dimensions, **transform_kwargs),
                self.num_repeat,
            )
        # extractor
        extractor_cfg_key = pipe_config.extractor_config_key
        extractor_unique_key = pipe_config.extractor_unique_key
        extractor_exists = extractor_unique_key in self.extractors
        if pipe_config.reuse_extractor and extractor_exists:
            extractors = self.extractors[extractor_unique_key]
        else:
            if extractor_exists:
                new_index = 0
                new_extractor_unique_key = extractor_unique_key
                while new_extractor_unique_key in self.extractors:
                    new_index += 1
                    new_extractor_unique_key = f"{extractor_unique_key}_{new_index}"
                extractor_unique_key = new_extractor_unique_key
            if extractor_cfg_key in self._extractor_configs:
                extractor_kwargs = self._extractor_configs[extractor_cfg_key]
            else:
                extractor_cfg = Configs.get(
                    pipe_config.extractor_scope,
                    pipe_config.extractor_config,
                    **extractor_config,
                )
                extractor_kwargs = extractor_cfg.pop()
                self._extractor_configs[extractor_cfg_key] = extractor_kwargs
            if pipe_config.use_extractor_meta:
                extractor_kwargs = extractor_kwargs[pipe_config.extractor]
            update_dict(meta_extractor_config, extractor_kwargs)
            extractors = _get_clones(
                ExtractorBase.make(
                    pipe_config.extractor,
                    transforms[0].out_dim,
                    transforms[0].dimensions,
                    extractor_kwargs,
                ),
                self.num_repeat,
            )
        # head
        head_cfg_key = pipe_config.head_config_key
        if head_cfg_key in self._head_configs:
            head_kwargs = self._head_configs[head_cfg_key]
            head_cfg = self._head_config_ins_dict[head_cfg_key]
            head_cfg.in_dim = extractors[0].out_dim
        else:
            head_cfg = HeadConfigs.get(
                pipe_config.head_scope,
                pipe_config.head_config,
                in_dim=extractors[0].out_dim,
                tr_data=self.tr_data,
                tr_weights=self.tr_weights,
                dimensions=self.dimensions,
                **head_config,
            )
            head_kwargs = head_cfg.pop()
            self._head_configs[head_cfg_key] = head_kwargs
            self._head_config_ins_dict[head_cfg_key] = head_cfg
        if pipe_config.use_head_meta:
            head_kwargs = head_kwargs[pipe_config.head]
        head_cfg.inject_dimensions(head_kwargs)
        update_dict(meta_head_config, head_kwargs)
        heads = _get_clones(
            HeadBase.make(pipe_config.head, head_kwargs),
            self.num_repeat,
        )
        # gather
        self.pipes[key] = transform_key, extractor_unique_key, head_cfg_key
        if not transform_exists:
            self.transforms[transform_key] = transforms
        self.extractors[extractor_unique_key] = extractors
        self.heads[key] = heads
        # bypass
        if transforms[0].out_dim == 0 or extractors[0].out_dim == 0:
            self.bypassed_pipes.add(key)

    # Inheritance

    @property
    def input_sample(self) -> tensor_dict_type:
        sample = next(iter(self.tr_loader))
        if self.tr_loader.return_indices:
            assert isinstance(sample, tuple)
            return sample[0]
        assert isinstance(sample, dict)
        return sample

    @property
    def output_probabilities(self) -> bool:
        return False

    def _preset_config(self) -> None:
        pass

    def _init_config(self) -> None:
        default_encoding_method = set()
        if self.registered_pipes is None:
            raise ValueError(f"No `pipe` is registered in {type(self).__name__}")
        for pipe_config in self.registered_pipes.values():
            transform_config = transform_config_mapping[pipe_config.transform]
            if transform_config["one_hot"]:
                default_encoding_method.add("one_hot")
            if transform_config["embedding"]:
                default_encoding_method.add("embedding")
        self.default_encoding_method = self.config.setdefault(
            "default_encoding_method", list(default_encoding_method)
        )
        self.loss_config["input_logits"] = not self.output_probabilities

    def forward(
        self,
        batch: tensor_dict_type,
        batch_idx: Optional[int] = None,
        state: Optional[TrainerState] = None,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        x_batch = batch["x_batch"]
        split = self._split_features(x_batch, batch_indices, loader_name)
        outputs = self.execute(split)
        # check whether outputs from each pipe are of identical type
        return_type = None
        for pipe_outputs in outputs.values():
            pipe_outputs_type = type(pipe_outputs)
            if return_type is None:
                return_type = pipe_outputs_type
            elif return_type is not pipe_outputs_type:
                raise ValueError(
                    f"some pipe(s) return `{return_type}` but "
                    f"other(s) return `{pipe_outputs_type}`"
                )
        # if return_type is Tensor, simply reduce them
        if return_type is torch.Tensor:
            return {"predictions": self.aggregator.reduce(outputs, **kwargs)}
        # otherwise, return_type should be dict, and all pipes should hold the same keys
        assert return_type is dict
        key_set = None
        for pipe_outputs in outputs.values():
            pipe_outputs_key_set = set(pipe_outputs)
            if key_set is None:
                key_set = pipe_outputs_key_set
            elif key_set != pipe_outputs_key_set:
                raise ValueError(
                    f"some pipe(s) return `{key_set}` but "
                    f"other(s) return `{pipe_outputs_key_set}`"
                )
        return {
            k: self.aggregator.reduce(
                {
                    pipe_key: pipe_outputs[k]
                    for pipe_key, pipe_outputs in outputs.items()
                },
                **kwargs,
            )
            for k in key_set  # type: ignore
        }

    def loss_function(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        batch_indices: Optional[torch.Tensor],
        forward_results: tensor_dict_type,
        state: Optional[TrainerState],
    ) -> tensor_dict_type:
        # requires returning `loss` key
        labels = batch[self.labels_key]
        # `sample_weights` could be accessed through:
        # 1) `self.tr_weights[batch_indices]` (for training)
        # 2) `self.cv_weights[batch_indices]` (for validation)
        forward_results = shallow_copy_dict(forward_results)
        forward_results["batch_indices"] = batch_indices
        losses = self.loss(forward_results, labels)
        if isinstance(losses, dict):
            return {k: v.mean() for k, v in losses.items()}
        return {"loss": losses.mean()}

    # API

    @property
    def configurations(self) -> Dict[str, Any]:
        return self.environment.config

    def _split_features(
        self,
        x_batch: Tensor,
        batch_indices: Optional[np.ndarray],
        loader_name: Optional[str],
    ) -> SplitFeatures:
        return self.dimensions.split_features(
            x_batch,
            batch_indices,
            loader_name,
            enable_timing=self.timing,
        )

    def _transform(
        self,
        transform: Transform,
        net: Union[Tensor, SplitFeatures],
    ) -> Tensor:
        return net if isinstance(net, Tensor) else transform(net)

    def _extract(
        self,
        extractor: ExtractorBase,
        transformed: Tensor,
        extract_kwargs: Dict[str, Any],
    ) -> Tensor:
        extracted = extractor(transformed, **extract_kwargs)
        extracted_shape = extracted.shape
        if extractor.flatten_ts:
            if len(extracted_shape) == 3:
                extracted = extracted.view(extracted_shape[0], -1)
        return extracted

    def _head(
        self,
        head: HeadBase,
        extracted: Tensor,
        head_kwargs: Dict[str, Any],
    ) -> Union[Tensor, tensor_dict_type]:
        return head(extracted, **head_kwargs)

    def execute(
        self,
        net: Union[Tensor, SplitFeatures],
        *,
        clear_cache: bool = True,
        extract_kwargs_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        head_kwargs_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> tensor_dict_type:
        results: Dict[str, Any] = {}
        for i in range(self.num_repeat):
            for key, (transform_key, extractor_key, _) in self.pipes.items():
                if key in self.bypassed_pipes:
                    continue
                # transform
                transformed = self._transform_cache.get(transform_key)
                if transformed is None:
                    transform = self.transforms[transform_key][i]
                    transformed = self._transform(transform, net)
                    self._transform_cache[transform_key] = transformed
                # extract
                if extract_kwargs_dict is None:
                    extract_kwargs_dict = {}
                extracted = self._extractor_cache.get(extractor_key)
                if extracted is None:
                    extract_kwargs = extract_kwargs_dict.get(extractor_key, {})
                    extracted = self._extract(
                        self.extractors[extractor_key][i],
                        transformed,
                        extract_kwargs,
                    )
                    self._extractor_cache[extractor_key] = extracted
                if head_kwargs_dict is None:
                    head_kwargs_dict = {}
                head_kwargs = head_kwargs_dict.get(key, {})
                head_result = self._head(self.heads[key][i], extracted, head_kwargs)
                if isinstance(head_result, Tensor):
                    results.setdefault(key, []).append(head_result)
                else:
                    key_results = results.setdefault(key, {})
                    for k, v in head_result.items():
                        key_results.setdefault(k, []).append(v)
            if clear_cache:
                self.clear_execute_cache()
        # aggregate num_repeat results
        for k in sorted(results):
            v = results[k]
            if isinstance(v, list):
                results[k] = torch.stack(v).mean(0)
            else:
                for vk in sorted(v):
                    v[vk] = torch.stack(v[vk]).mean(0)
        return results

    def clear_execute_cache(self) -> None:
        self._transform_cache = {}
        self._extractor_cache = {}

    def get_split(self, processed: np.ndarray, device: torch.device) -> SplitFeatures:
        with torch.no_grad():
            return self._split_features(to_torch(processed).to(device), None, None)

    def export_context(self) -> context_error_handler:
        class _(context_error_handler):
            def __init__(self, model: ModelBase):
                self.fast_dndf_settings: Dict[DNDF, bool] = {}

                def _inject(node: Module) -> None:
                    for child in node.children():
                        if isinstance(child, DNDF):
                            self.fast_dndf_settings[child] = child._fast
                        elif isinstance(child, Module):
                            _inject(child)

                _inject(model)

            def __enter__(self) -> None:
                for dndf in self.fast_dndf_settings:
                    dndf._fast = False

            def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                for dndf, fast in self.fast_dndf_settings.items():
                    dndf._fast = fast

        return _(self)

    def extra_repr(self) -> str:
        pipe_str = "\n".join(
            [f"  ({key}): {' -> '.join(pipe[1:])}" for key, pipe in self.pipes.items()]
        )
        return f"(pipes): Pipes(\n{pipe_str}\n)"

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global model_dict

        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, model_dict, before_register=before)

    @classmethod
    def register_pipe(
        cls,
        key: str,
        *,
        transform: str = "default",
        extractor: Optional[str] = None,
        reuse_extractor: bool = True,
        head: Optional[str] = None,
        extractor_config: str = "default",
        head_config: str = "default",
        extractor_meta_scope: Optional[str] = None,
        head_meta_scope: Optional[str] = None,
    ) -> Callable[[Type], Type]:
        if head is None:
            head = key
        elif extractor is None:
            extractor = key
        if extractor is None:
            extractor = "identity"

        def _core(cls_: Type) -> Type:
            assert head is not None
            assert extractor is not None
            cfg = PipeConfig(
                transform,
                extractor,
                reuse_extractor,
                head,
                extractor_config,
                head_config,
                extractor_meta_scope,
                head_meta_scope,
            )
            if cls_.registered_pipes is None:
                cls_.registered_pipes = {key: cfg}
            else:
                cls_.registered_pipes[key] = cfg
            return cls_

        return _core


class SiameseModelBase(ModelBase):
    def _extract(
        self,
        extractor: ExtractorBase,
        transformed: Tensor,
        extract_kwargs: Dict[str, Any],
    ) -> Tensor:
        if not self.training:
            return super()._extract(extractor, transformed, extract_kwargs)
        num_slice = transformed.shape[0] // 2
        t1, t2 = transformed[:num_slice], transformed[num_slice : 2 * num_slice]
        e1 = super()._extract(extractor, t1, shallow_copy_dict(extract_kwargs))
        e2 = super()._extract(extractor, t2, shallow_copy_dict(extract_kwargs))
        return e2 - e1

    def forward(
        self,
        batch: tensor_dict_type,
        batch_idx: Optional[int] = None,
        state: Optional[TrainerState] = None,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        results = super().forward(
            batch,
            batch_idx,
            state,
            batch_indices,
            loader_name,
            **kwargs,
        )
        if not self.training:
            return results
        labels = batch["labels"]
        num_slice = labels.shape[0] // 2
        l1, l2 = labels[:num_slice], labels[num_slice : 2 * num_slice]
        batch["labels"] = l2 - l1
        return results


__all__ = [
    "ModelBase",
    "SiameseModelBase",
]
