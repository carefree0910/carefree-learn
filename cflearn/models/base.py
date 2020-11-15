import torch
import pprint

import numpy as np
import torch.nn as nn

from typing import *
from abc import ABCMeta
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleDict
from torch.optim import Optimizer
from cfdata.tabular import ColumnTypes
from cfdata.tabular import DataLoader
from cftool.misc import register_core
from cftool.misc import LoggingMixin

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None

from ..losses import *
from ..modules import *
from ..types import tensor_dict_type
from ..misc.configs import Configs
from ..misc.toolkit import to_torch
from ..modules.heads import HeadBase
from ..modules.heads import HeadConfigs
from ..modules.transform import Transform
from ..modules.transform import Dimensions
from ..modules.transform import SplitFeatures
from ..modules.extractors import ExtractorBase

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
    def head_config_key(self) -> str:
        if self.head_meta_scope is None:
            prefix = self.head
        else:
            prefix = self.head_meta_scope
        return f"{prefix}_{self.head_config}"


class ModelBase(Module, LoggingMixin, metaclass=ABCMeta):
    registered_pipes: Optional[Dict[str, PipeConfig]] = None

    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        tr_loader: DataLoader,
        cv_loader: DataLoader,
        tr_weights: Optional[np.ndarray],
        cv_weights: Optional[np.ndarray],
        device: torch.device,
        *,
        use_tqdm: bool,
    ):
        super().__init__()
        self.ema: Optional[EMA] = None
        self.device = device
        self.use_tqdm = use_tqdm
        self._pipeline_config = pipeline_config
        self.timing = self._pipeline_config["use_timing_context"]
        self.config = pipeline_config.setdefault("model_config", {})
        self.tr_loader = tr_loader
        self.cv_loader = cv_loader
        self.tr_data = tr_loader.data
        self.cv_data = None if cv_loader is None else cv_loader.data
        self.tr_weights, self.cv_weights = tr_weights, cv_weights
        self._preset_config()
        self._init_config()
        self._init_loss()
        # encoder
        excluded = 0
        numerical_columns_mapping = {}
        categorical_columns_mapping = {}
        categorical_dims = []
        encoding_methods = []
        encoding_configs = []
        true_categorical_columns = []
        if self.tr_data.is_simplify:
            for idx in range(self.tr_data.raw_dim):
                numerical_columns_mapping[idx] = idx
        else:
            ts_indices = self.tr_data.ts_indices
            recognizers = self.tr_data.recognizers
            sorted_indices = [idx for idx in sorted(recognizers) if idx != -1]
            for idx in sorted_indices:
                recognizer = recognizers[idx]
                if not recognizer.info.is_valid or idx in ts_indices:
                    excluded += 1
                elif recognizer.info.column_type is ColumnTypes.NUMERICAL:
                    numerical_columns_mapping[idx] = idx - excluded
                else:
                    str_idx = str(idx)
                    categorical_dims.append(
                        self.tr_data.recognizers[idx].num_unique_values
                    )
                    encoding_methods.append(
                        self._encoding_methods.setdefault(
                            str_idx, self._default_encoding_method
                        )
                    )
                    encoding_configs.append(
                        self._encoding_configs.setdefault(
                            str_idx, self._default_encoding_configs
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
        self._bypass_info_dict: Dict[str, Union[bool, Dict[str, bool]]] = {}
        if self.registered_pipes is None:
            raise ValueError(f"No `pipe` is registered in {type(self).__name__}")
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
        # caches
        self._transform_cache: Dict[str, Tensor] = {}
        self._extractor_cache: Dict[str, Tensor] = {}

    @property
    def num_history(self) -> int:
        num_history = 1
        if self.tr_data.is_ts:
            sampler_config = self._pipeline_config["sampler_config"]
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
        # transform
        transform_cfg = Configs.get(
            "transform",
            pipe_config.transform,
            **transform_config,
        )
        transform = Transform(self.dimensions, **transform_cfg.pop())
        # extractor
        extractor_cfg_key = pipe_config.extractor_config_key
        extractor_exists = extractor_cfg_key in self.extractors
        if pipe_config.reuse_extractor and extractor_exists:
            extractor = self.extractors[extractor_cfg_key]
        else:
            if extractor_exists:
                new_index = 1
                new_extractor_cfg_key = extractor_cfg_key
                while new_extractor_cfg_key in self.extractors:
                    new_extractor_cfg_key = f"{extractor_cfg_key}_{new_index}"
                extractor_cfg_key = new_extractor_cfg_key
            if extractor_cfg_key in self._extractor_configs:
                extractor_config = self._extractor_configs[extractor_cfg_key]
            else:
                extractor_cfg = Configs.get(
                    pipe_config.extractor_scope,
                    pipe_config.extractor_config,
                    **extractor_config,
                )
                extractor_config = extractor_cfg.pop()
                self._extractor_configs[extractor_cfg_key] = extractor_config
            if pipe_config.use_extractor_meta:
                extractor_config = extractor_config[pipe_config.extractor]
            extractor = ExtractorBase.make(
                pipe_config.extractor,
                transform.out_dim,
                transform.dimensions,
                extractor_config,
            )
        head_cfg_key = pipe_config.head_config_key
        if head_cfg_key in self._head_configs:
            head_config = self._head_configs[head_cfg_key]
            head_cfg = self._head_config_ins_dict[head_cfg_key]
            head_cfg.in_dim = extractor.out_dim
            bypass_info = self._bypass_info_dict[head_cfg_key]
        else:
            head_cfg = HeadConfigs.get(
                pipe_config.head_scope,
                pipe_config.head_config,
                in_dim=extractor.out_dim,
                tr_data=self.tr_data,
                tr_weights=self.tr_weights,
                dimensions=self.dimensions,
                **head_config,
            )
            head_config = head_cfg.pop()
            bypass_info = head_cfg.should_bypass(head_config)
            self._head_configs[head_cfg_key] = head_config
            self._head_config_ins_dict[head_cfg_key] = head_cfg
            self._bypass_info_dict[head_cfg_key] = bypass_info
        if not pipe_config.use_head_meta:
            should_bypass = bypass_info
        else:
            assert isinstance(bypass_info, dict)
            head_config = head_config[pipe_config.head]
            should_bypass = bypass_info[pipe_config.head]
        head_cfg.inject_dimensions(head_config)
        head = HeadBase.make(pipe_config.head, head_config)
        # gather
        self.pipes[key] = transform_cfg.name, extractor_cfg_key, head_cfg_key
        self.transforms[transform_cfg.name] = transform
        self.extractors[extractor_cfg_key] = extractor
        self.heads[key] = head
        if should_bypass:
            self.bypassed_pipes.add(key)

    # Inheritance

    @property
    def input_sample(self) -> tensor_dict_type:
        x = self.tr_data.processed.x[:2]
        y = self.tr_data.processed.y[:2]
        x, y = map(to_torch, [x, y])
        return {"x_batch": x, "y_batch": y}

    @property
    def output_probabilities(self) -> bool:
        return False

    def _preset_config(self) -> None:
        pass

    def _init_config(self) -> None:
        # encoding
        encoding_methods = self.config.setdefault("encoding_methods", {})
        encoding_configs = self.config.setdefault("encoding_configs", {})
        self._encoding_methods = {str(k): v for k, v in encoding_methods.items()}
        self._encoding_configs = {str(k): v for k, v in encoding_configs.items()}
        self._default_encoding_configs = self.config.setdefault(
            "default_encoding_configs", {}
        )
        self._default_encoding_method = self.config.setdefault(
            "default_encoding_method", "embedding"
        )
        # loss
        self._loss_config = self.config.setdefault("loss_config", {})
        self._loss_config["input_logits"] = not self.output_probabilities

    def _init_loss(self) -> None:
        if self.tr_data.is_reg:
            self.loss: Module = nn.L1Loss(reduction="none")
        else:
            self.loss = FocalLoss(self._loss_config, reduction="none")

    def merge_outputs(
        self,
        outputs: tensor_dict_type,
        **kwargs: Any,
    ) -> tensor_dict_type:
        # requires returning `predictions` key
        values = list(outputs.values())
        output = None
        for value in values:
            if value is None:
                continue
            if output is None:
                output = value
            else:
                output = output + value
        return {"predictions": output}

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        batch_step: int = 0,
        **kwargs: Any,
    ) -> tensor_dict_type:
        # batch will have `categorical`, `numerical` and `labels` keys
        x_batch = batch["x_batch"]
        split = self._split_features(x_batch, batch_indices, loader_name)
        outputs = self.execute(split)
        return self.merge_outputs(outputs, **kwargs)

    def loss_function(
        self,
        batch: tensor_dict_type,
        batch_indices: np.ndarray,
        forward_results: tensor_dict_type,
        batch_step: int,
    ) -> tensor_dict_type:
        # requires returning `loss` key
        y_batch = batch["y_batch"]
        if self.tr_data.is_clf:
            y_batch = y_batch.view(-1)
        predictions = forward_results["predictions"]
        # `sample_weights` could be accessed through:
        # 1) `self.tr_weights[batch_indices]` (for training)
        # 2) `self.cv_weights[batch_indices]` (for validation)
        losses = self.loss(predictions, y_batch)
        return {"loss": losses.mean()}

    # API

    @property
    def use_ema(self) -> bool:
        return self.ema is not None

    def init_ema(self) -> None:
        ema_decay = self.config.setdefault("ema_decay", 0.0)
        if 0.0 < ema_decay < 1.0:
            named_params = list(self.named_parameters())
            self.ema = EMA(ema_decay, named_params)  # type: ignore

    def apply_ema(self) -> None:
        if self.ema is None:
            raise ValueError("`ema` is not defined")
        self.ema()

    def info(self, *, return_only: bool = False) -> str:
        msg = "\n".join(["=" * 100, "configurations", "-" * 100, ""])
        msg += (
            pprint.pformat(self._pipeline_config, compact=True)
            + "\n"
            + "-" * 100
            + "\n"
        )
        msg += "\n".join(["=" * 100, "parameters", "-" * 100, ""])
        for name, param in self.named_parameters():
            if param.requires_grad:
                msg += name + "\n"
        msg += "\n".join(["-" * 100, "=" * 100, "buffers", "-" * 100, ""])
        for name, param in self.named_buffers():
            msg += name + "\n"
        msg += "\n".join(
            ["-" * 100, "=" * 100, "structure", "-" * 100, str(self), "-" * 100, ""]
        )
        if not return_only:
            self.log_block_msg(msg, verbose_level=4)  # type: ignore
        all_msg, msg = msg, "=" * 100 + "\n"
        n_tr = len(self.tr_data)
        n_cv = None if self.cv_data is None else len(self.cv_data)
        msg += f"{self.info_prefix}training data : {n_tr}\n"
        msg += f"{self.info_prefix}valid    data : {n_cv}\n"
        msg += "-" * 100
        if not return_only:
            self.log_block_msg(msg, verbose_level=3)  # type: ignore
        return "\n".join([all_msg, msg])

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

    def execute(
        self,
        net: Union[Tensor, SplitFeatures],
        *,
        clear_cache: bool = True,
        extract_kwargs_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        head_kwargs_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> tensor_dict_type:
        results: Dict[str, Tensor] = {}
        for key, (transform_key, executor_key, _) in self.pipes.items():
            if key in self.bypassed_pipes:
                continue
            # transform
            transformed = self._transform_cache.get(transform_key)
            if transformed is None:
                transform = self.transforms[transform_key]
                transformed = net if isinstance(net, Tensor) else transform(net)
                self._transform_cache[transform_key] = transformed
            # extract
            if extract_kwargs_dict is None:
                extract_kwargs_dict = {}
            extracted = self._extractor_cache.get(executor_key)
            if extracted is None:
                extractor = self.extractors[executor_key]
                extract_kwargs = extract_kwargs_dict.get(executor_key, {})
                extracted = extractor(transformed, **extract_kwargs)
                extracted_shape = extracted.shape
                if extractor.flatten_ts:
                    if len(extracted_shape) == 3:
                        extracted = extracted.view(extracted_shape[0], -1)
                self._extractor_cache[executor_key] = extracted
            # execute
            if head_kwargs_dict is None:
                head_kwargs_dict = {}
            head_kwargs = head_kwargs_dict.get(key, {})
            results[key] = self.heads[key](extracted, **head_kwargs)
        # finalize
        if clear_cache:
            self.clear_execute_cache()
        return results

    def clear_execute_cache(self) -> None:
        self._transform_cache = {}
        self._extractor_cache = {}

    def _optimizer_step(
        self,
        optimizers: Dict[str, Optimizer],
        grad_scalar: Optional["amp.GradScaler"],  # type: ignore
    ) -> None:
        for opt in optimizers.values():
            if grad_scalar is None:
                opt.step()
            else:
                grad_scalar.step(opt)
                grad_scalar.update()
            opt.zero_grad()

    def get_split(self, processed: np.ndarray, device: torch.device) -> SplitFeatures:
        return self._split_features(torch.from_numpy(processed).to(device), None, None)

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


__all__ = ["ModelBase"]
