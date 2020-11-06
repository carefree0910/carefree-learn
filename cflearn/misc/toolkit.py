import math
import torch
import logging

import numpy as np
import torch.nn as nn

from typing import *
from abc import abstractmethod
from abc import ABCMeta
from functools import partial
from cftool.misc import context_error_handler
from cftool.misc import Incrementer
from cftool.misc import LoggingMixin
from cfdata.types import np_int_type
from cfdata.types import np_float_type

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None

from ..types import data_type
from ..types import param_type
from ..types import np_dict_type
from ..types import tensor_dict_type


def is_int(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.integer)


def is_float(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.floating)


def to_standard(arr: np.ndarray) -> np.ndarray:
    if is_int(arr):
        arr = arr.astype(np_int_type)
    elif is_float(arr):
        arr = arr.astype(np_float_type)
    return arr


def to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(to_standard(arr))


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def to_2d(arr: data_type) -> data_type:
    if arr is None or isinstance(arr, str):
        return None
    if isinstance(arr, np.ndarray):
        return arr.reshape([len(arr), -1])
    if isinstance(arr[0], list):
        return arr
    return [[elem] for elem in arr]


def to_prob(raw: np.ndarray) -> np.ndarray:
    return nn.functional.softmax(torch.from_numpy(raw), dim=1).numpy()


def collate_np_dicts(ds: List[np_dict_type], axis: int = 0) -> np_dict_type:
    results = {}
    d0 = ds[0]
    for k in d0.keys():
        if not isinstance(d0[k], np.ndarray):
            continue
        arrays = []
        for rs in ds:
            array = rs[k]
            if len(array.shape) == 0:
                array = array.reshape([1])
            arrays.append(array)
        results[k] = np.concatenate(arrays, axis=axis)
    return results


def collate_tensor_dicts(ds: List[tensor_dict_type], dim: int = 0) -> tensor_dict_type:
    results = {}
    d0 = ds[0]
    for k in d0.keys():
        if not isinstance(d0[k], torch.Tensor):
            continue
        tensors = []
        for rs in ds:
            tensor = rs[k]
            if len(tensor.shape) == 0:
                tensor = tensor.view([1])
            tensors.append(tensor)
        results[k] = torch.cat(tensors, dim=dim)
    return results


def switch_requires_grad(params: List[nn.Parameter], requires_grad: bool) -> None:
    for param in params:
        param.requires_grad_(requires_grad)


def get_gradient(
    y: torch.Tensor,
    x: torch.Tensor,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    grads = torch.autograd.grad(y, x, torch.ones_like(y), retain_graph, create_graph)
    if len(grads) == 1:
        return grads[0]
    return grads


class Initializer(LoggingMixin):
    """
    Initializer for neural network weights

    Examples
    --------
    >>> initializer = Initializer({})
    >>> linear = nn.Linear(10, 10)
    >>> initializer.xavier_uniform(linear.weight)

    """

    defined_initialization = {
        "xavier_uniform",
        "xavier_normal",
        "normal",
        "truncated_normal",
    }
    custom_initializer: Dict[str, Callable] = {}

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._verbose_level = config.setdefault("verbose_level", 2)

    def initialize(self, param: param_type, method: str) -> Any:
        custom_initializer = self.custom_initializer.get(method)
        if custom_initializer is None:
            return getattr(self, method)(param)
        return custom_initializer(self, param)

    @classmethod
    def add_initializer(cls, f: Callable, name: str) -> None:
        if name in cls.defined_initialization:
            print(f"{cls.warning_prefix}'{name}' initializer is already defined")
            return
        cls.defined_initialization.add(name)
        cls.custom_initializer[name] = f

    def xavier_uniform(self, param: param_type) -> None:
        gain = self.config.setdefault("gain", 1.0)
        nn.init.xavier_uniform_(param.data, gain)

    def xavier_normal(self, param: param_type) -> None:
        gain = self.config.setdefault("gain", 1.0)
        nn.init.xavier_normal_(param.data, gain)

    def normal(self, param: param_type) -> None:
        mean = self.config.setdefault("mean", 0.0)
        std = self.config.setdefault("std", 1.0)
        with torch.no_grad():
            param.data.normal_(mean, std)

    def truncated_normal(self, param: param_type) -> None:
        span = self.config.setdefault("span", 2.0)
        mean = self.config.setdefault("mean", 0.0)
        std = self.config.setdefault("std", 1.0)
        tol = self.config.setdefault("tol", 0.0)
        epoch = self.config.setdefault("epoch", 20)
        num_elem = param.numel()
        weight_base = param.new_empty(num_elem).normal_()
        get_invalid = lambda w: (w > span) | (w < -span)
        invalid = get_invalid(weight_base)
        success = False
        for _ in range(epoch):
            num_invalid = invalid.sum().item()
            if num_invalid / num_elem <= tol:
                success = True
                break
            with torch.no_grad():
                weight_base[invalid] = param.new_empty(num_invalid).normal_()
                invalid = get_invalid(weight_base)
        if not success:
            self.log_msg(
                f"invalid ratio for truncated normal : {invalid.to(torch.float32).mean():8.6f}, "
                f"it might cause by too little epoch ({epoch}) or too small tolerance ({tol})",
                prefix=self.warning_prefix,
                verbose_level=2,
                msg_level=logging.WARNING,
            )
        with torch.no_grad():
            param.data.copy_(weight_base.reshape(param.shape))
            param.data.mul_(std).add_(mean)


class _multiplied_activation(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        ratio: float,
        trainable: bool = True,
    ):
        super().__init__()
        self.trainable = trainable
        ratio_ = torch.tensor([ratio], dtype=torch.float32)
        self.ratio = ratio_ if not trainable else nn.Parameter(ratio_)

    @abstractmethod
    def _core(self, multiplied: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._core(x * self.ratio)

    def extra_repr(self) -> str:
        return f"ratio={self.ratio.item()}, trainable={self.trainable}"


class Lambda(nn.Module):
    def __init__(self, fn: Callable, name: str = None):
        super().__init__()
        self.name = name
        self.fn = fn

    def extra_repr(self) -> str:
        return "" if self.name is None else self.name

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


class Activations:
    """
    Wrapper class for pytorch activations
    * when pytorch implemented corresponding activation, it will be returned
    * otherwise, custom implementation will be returned

    Parameters
    ----------
    configs : {None, dict}, configuration for the activation

    Examples
    --------
    >>> act = Activations()
    >>> print(type(act.ReLU))  # <class 'nn.modules.activation.ReLU'>
    >>> print(type(act.module("ReLU")))  # <class 'nn.modules.activation.ReLU'>
    >>> print(type(act.Tanh))  # <class 'nn.modules.activation.Tanh'>
    >>> print(type(act.one_hot))  # <class '__main__.Activations.one_hot.<locals>.OneHot'>

    """

    def __init__(self, configs: Optional[Dict[str, Any]] = None):
        if configs is None:
            configs = {}
        self.configs = configs

    def __getattr__(self, item: str) -> nn.Module:
        kwargs = self.configs.setdefault(item, {})
        try:
            return getattr(nn, item)(**kwargs)
        except AttributeError:
            func = getattr(torch, item, None)
            if func is None:
                raise NotImplementedError(
                    "neither pytorch nor custom Activations "
                    f"implemented activation '{item}'"
                )
            return Lambda(partial(func, **kwargs), item)

    def module(self, name: str) -> nn.Module:
        if name is None:
            return nn.Identity()
        return getattr(self, name)

    # publications

    @property
    def glu(self) -> nn.Module:
        config = self.configs.setdefault("glu", {})
        in_dim = config.get("in_dim")
        if in_dim is None:
            raise ValueError("`in_dim` should be provided in glu")
        bias = config.setdefault("bias", True)

        class GLU(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(in_dim, 2 * in_dim, bias)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                projection, gate = self.linear(x).chunk(2, dim=1)
                return projection * torch.sigmoid(gate)

        return GLU()

    @property
    def mish(self) -> nn.Module:
        class Mish(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * (torch.tanh(nn.functional.softplus(x)))

        return Mish()

    # custom

    # TODO : After updated to pytorch>=1.7.0, re-implement this
    @property
    def logit(self) -> nn.Module:
        kwargs = self.configs.setdefault("logit", {})
        eps = kwargs.setdefault("eps", 1.0e-6)

        def _logit(x: torch.Tensor) -> torch.Tensor:
            x = torch.clamp(x, eps, 1.0 - eps)
            return torch.log(x / (1.0 - x))

        return Lambda(_logit, f"logit_{eps:.2e}")

    @property
    def atanh(self) -> nn.Module:
        kwargs = self.configs.setdefault("atanh", {})
        eps = kwargs.setdefault("eps", 1.0e-6)

        def _atanh(x: torch.Tensor) -> torch.Tensor:
            return torch.atanh(torch.clamp(x, -1.0 + eps, 1.0 - eps))

        return Lambda(_atanh, f"atanh_{eps:.2e}")

    @property
    def isoftplus(self) -> nn.Module:
        kwargs = self.configs.setdefault("isoftplus", {})
        eps = kwargs.setdefault("eps", 1.0e-6)

        def _isoftplus(x: torch.Tensor) -> torch.Tensor:
            return torch.log(x.clamp_min(eps).exp() - 1.0)

        return Lambda(_isoftplus, f"isoftplus_{eps:.2e}")

    @property
    def sign(self) -> nn.Module:
        return Lambda(lambda x: torch.sign(x), "sign")

    @property
    def one_hot(self) -> nn.Module:
        f = lambda x: x * (x == torch.max(x, dim=1, keepdim=True)[0]).to(torch.float32)
        return Lambda(f, "one_hot")

    @property
    def sine(self) -> nn.Module:
        return Lambda(lambda x: torch.sin(x), "sine")

    @property
    def multiplied_sine(self) -> nn.Module:
        class MultipliedSine(_multiplied_activation):
            def _core(self, multiplied: torch.Tensor) -> torch.Tensor:
                return torch.sin(multiplied)

        config = self.configs.setdefault("multiplied_sine", {})
        config.setdefault("ratio", 10.0)
        return MultipliedSine(**config)

    @property
    def multiplied_tanh(self) -> nn.Module:
        class MultipliedTanh(_multiplied_activation):
            def _core(self, multiplied: torch.Tensor) -> torch.Tensor:
                return torch.tanh(multiplied)

        return MultipliedTanh(**self.configs.setdefault("multiplied_tanh", {}))

    @property
    def multiplied_softmax(self) -> nn.Module:
        class MultipliedSoftmax(_multiplied_activation):
            def __init__(self, ratio: float, dim: int = 1, trainable: bool = True):
                super().__init__(ratio, trainable)
                self.dim = dim

            def _core(self, multiplied: torch.Tensor) -> torch.Tensor:
                return nn.functional.softmax(multiplied, dim=self.dim)

        return MultipliedSoftmax(**self.configs.setdefault("multiplied_softmax", {}))

    @classmethod
    def make(
        cls,
        name: Optional[str],
        config: Optional[Dict[str, Any]],
    ) -> nn.Module:
        if name is None:
            return nn.Identity()
        if config is None:
            config = {}
        if name.startswith("leaky_relu"):
            splits = name.split("_")
            if len(splits) == 3:
                config["negative_slope"] = float(splits[-1])
            config.setdefault("inplace", True)
            return nn.LeakyReLU(**config)
        if name.lower() == "relu":
            name = "ReLU"
            config.setdefault("inplace", True)
        return cls({name: config}).module(name)


class TrainMonitor:
    """
    Util class to monitor training process of a neural network
    * If overfitting, it will tell the model to early-stop
    * If underfitting, it will tell the model to extend training process
    * If better performance acquired, it will tell the model to save a checkpoint
    * If performance sticks on a plateau, it will tell the model to stop training (to save time)

    Warnings
    ----------
    * Performance MUST be evaluated on the cross validation dataset instead of the training set if possible
    * `register_trainer` method MUST be called before monitoring
    * instance passed to`register_trainer` method MUST be a subclass of `LoggingMixin`,
      and must include `_epoch_count`, `num_epoch`, `max_epoch` attributes and
      `save_checkpoint` method

    Parameters
    ----------
    sign : {-1, 1}
        * -1 : the metric will be the lower the better (e.g. mae, mse)
        * 1  : the metric will be the higher the better (e.g. auc, acc)
    num_scores_per_snapshot : int, indicates snapshot frequency
        * `TrainMonitor` will perform a snapshot every `num_scores_per_snapshot` scores are recorded
    history_ratio : float, indicates the ratio of the history's window width
        * history window width will be `num_scores_per_snapshot` * `history_ratio`
    tolerance_ratio : float, indicates the ratio of tolerance
        * tolerance base will be `num_scores_per_snapshot` * `tolerance_ratio`
        * judgements of 'overfitting' and 'performance sticks on a plateau' will based on 'tolerance base'
    extension : int, indicates how much epoch to extend when underfitting occurs
    std_floor : float, indicates the floor of history's std used for judgements
    std_ceiling : float, indicates the ceiling of history's std used for judgements
    aggressive : bool, indicates the strategy of monitoring
        * True  : it will tell the model to save every checkpoint when better metric is reached
        * False : it will be more careful since better metric may lead to
        more seriously over-fitting on cross validation set

    Examples
    ----------
    >>> from cftool.ml import Metrics
    >>>
    >>> x, y, model = ...
    >>> metric = Metrics("mae")
    >>> monitor = TrainMonitor(metric.sign).register_trainer(model)
    >>> n_epoch, epoch_count = 20, 0
    >>> while epoch_count <= n_epoch:
    >>>     model.train()
    >>>     predictions = model.predict(x)
    >>>     if monitor.check_terminate(metric.metric(y, predictions)):
    >>>         break

    """

    def __init__(
        self,
        sign: int,
        num_scores_per_snapshot: int = 1,
        history_ratio: int = 3,
        tolerance_ratio: int = 2,
        extension: int = 5,
        std_floor: float = 0.001,
        std_ceiling: float = 0.01,
        aggressive: bool = False,
    ):
        self.sign = sign
        self.num_scores_per_snapshot = num_scores_per_snapshot
        self.num_history = int(num_scores_per_snapshot * history_ratio)
        self.num_tolerance = int(num_scores_per_snapshot * tolerance_ratio)
        self.extension = extension
        self.is_aggressive = aggressive
        self.std_floor, self.std_ceiling = std_floor, std_ceiling
        self._scores: List[float] = []
        self.plateau_flag = False
        self._is_best: Optional[bool] = None
        self._running_best: Optional[float] = None
        self._descend_increment = self.num_history * extension / 30.0
        self._incrementer = Incrementer(self.num_history)

        self._over_fit_performance = math.inf
        self._best_checkpoint_performance = -math.inf
        self._descend_counter = self._plateau_counter = self.over_fitting_flag = 0.0
        self.info: Dict[str, Any] = {
            "terminate": False,
            "save_checkpoint": False,
            "save_best": aggressive,
            "info": None,
        }

    @property
    def log_msg(self) -> Callable:
        return self._trainer.log_msg

    @property
    def plateau_threshold(self) -> int:
        return 6 * self.num_tolerance * self.num_history

    def _update_running_info(self, last_score: float) -> float:
        self._incrementer.update(last_score)
        if self._running_best is None:
            if self._scores[0] > self._scores[1]:
                improvement = 0.0
                self._running_best, self._is_best = self._scores[0], False
            else:
                improvement = self._scores[1] - self._scores[0]
                self._running_best, self._is_best = self._scores[1], True
        elif self._running_best > last_score:
            improvement = 0
            self._is_best = False
        else:
            improvement = last_score - self._running_best
            self._running_best = last_score
            self._is_best = True
        return improvement

    def _handle_overfitting(self, last_score: float, res: float, std: float) -> None:
        if self._descend_counter == 0.0:
            self.info["save_best"] = True
            self._over_fit_performance = last_score
        self._descend_counter += min(self.num_tolerance / 3, -res / std)
        self.log_msg(
            f"descend counter updated : {self._descend_counter:6.4f}",
            prefix=self._trainer.info_prefix,
            verbose_level=6,
            msg_level=logging.DEBUG,
        )
        self.over_fitting_flag = 1

    def _handle_recovering(
        self,
        improvement: float,
        last_score: float,
        res: float,
        std: float,
    ) -> None:
        if res > 3 * std and self._is_best and improvement > std:
            self.info["save_best"] = True
        new_counter = self._descend_counter - res / std
        if self._descend_counter > 0 >= new_counter:
            self._over_fit_performance = math.inf
            if last_score > self._best_checkpoint_performance:
                self._best_checkpoint_performance = last_score
                assert self._running_best is not None
                if last_score > self._running_best - std:
                    self._plateau_counter //= 2
                    self.info["save_checkpoint"] = True
                    self.info["info"] = (
                        f"current snapshot ({len(self._scores)}) seems to be working well, "
                        "saving checkpoint in case we need to restore"
                    )
            self.over_fitting_flag = 0
        if self._descend_counter > 0:
            self._descend_counter = max(new_counter, 0)
            self.log_msg(
                f"descend counter updated : {self._descend_counter:6.4f}",
                prefix=self._trainer.info_prefix,
                verbose_level=6,
                msg_level=logging.DEBUG,
            )

    def _handle_is_best(self) -> None:
        if self._is_best:
            self.info["terminate"] = False
            if self.info["save_best"]:
                self._plateau_counter //= 2
                self.info["save_checkpoint"] = True
                self.info["save_best"] = self.is_aggressive
                self.info["info"] = (
                    f"current snapshot ({len(self._scores)}) leads to best result we've ever had, "
                    "saving checkpoint since "
                )
                if self.over_fitting_flag:
                    self.info["info"] += "we've suffered from over-fitting"
                else:
                    self.info["info"] += "performance has improved significantly"

    def _handle_period(self, last_score: float) -> None:
        if self.is_aggressive:
            return
        if (
            len(self._scores) % self.num_scores_per_snapshot == 0
            and last_score > self._best_checkpoint_performance
        ):
            self._best_checkpoint_performance = last_score
            self._plateau_counter //= 2
            self.info["terminate"] = False
            self.info["save_checkpoint"] = True
            self.info["info"] = (
                f"current snapshot ({len(self._scores)}) leads to best checkpoint we've ever had, "
                "saving checkpoint in case we need to restore"
            )

    def _punish_extension(self) -> None:
        self.plateau_flag = True
        self._descend_counter += self._descend_increment

    def _handle_trainer_terminate(self, score: float) -> bool:
        trainer = self._trainer
        if self.info["terminate"]:
            self.log_msg(
                f"early stopped at n_epoch={trainer._epoch_count} due to '{self.info['info']}'",
                prefix=trainer.info_prefix,
            )
            return True
        if self.info["save_checkpoint"]:
            self.log_msg(f"{self.info['info']}", trainer.info_prefix, 3)
            trainer.save_checkpoint(score)
        if (
            trainer._epoch_count == trainer.num_epoch
            and trainer._epoch_count < trainer.max_epoch
            and not self.info["terminate"]
        ):
            self._punish_extension()
            new_epoch = trainer.num_epoch + self.extension
            trainer.num_epoch = min(new_epoch, trainer.max_epoch)
            self.log_msg(
                f"extending num_epoch to {trainer.num_epoch}",
                prefix=trainer.info_prefix,
                verbose_level=3,
            )
        if trainer._epoch_count == trainer.max_epoch:
            if not self.info["terminate"]:
                self.log_msg(
                    "model seems to be under-fitting but max_epoch reached, "
                    "increasing max_epoch may improve performance.",
                    trainer.info_prefix,
                )
            return True
        return False

    def register_trainer(self, trainer: Any) -> "TrainMonitor":
        self._trainer = trainer
        return self

    def check_terminate(self, new_metric: float) -> bool:
        last_score = new_metric * self.sign
        self._scores.append(last_score)
        n_history = min(self.num_history, len(self._scores))
        if math.isnan(new_metric):
            self.info["terminate"] = True
            self.info["info"] = "nan metric encountered"
        elif n_history != 1:
            improvement = self._update_running_info(last_score)
            self.info["save_checkpoint"] = False
            mean, std = self._incrementer.mean, self._incrementer.std
            std = min(std, self.std_ceiling)
            plateau_updated = False
            if std < self.std_floor:
                if self.plateau_flag:
                    increment = self.std_floor / max(std, self.std_floor / 6)
                    self._plateau_counter += increment
                    plateau_updated = True
            else:
                if self._plateau_counter > 0:
                    self._plateau_counter = max(self._plateau_counter - 1, 0)
                    plateau_updated = True
                res = last_score - mean
                if res < -std and last_score < self._over_fit_performance - std:
                    self._handle_overfitting(last_score, res, std)
                elif res > std:
                    self._handle_recovering(improvement, last_score, res, std)
            if plateau_updated:
                self.log_msg(
                    f"plateau counter updated : {self._plateau_counter:>6.4f} "
                    f"/ {self.plateau_threshold}",
                    prefix=self._trainer.info_prefix,
                    verbose_level=6,
                    msg_level=logging.DEBUG,
                )
            if self._plateau_counter >= self.plateau_threshold:
                self.info["info"] = "performance not improving"
                self.info["terminate"] = True
            else:
                if self._descend_counter >= self.num_tolerance:
                    self.info["info"] = "over-fitting"
                    self.info["terminate"] = True
                else:
                    self._handle_is_best()
                    self._handle_period(last_score)
                    if self.info["save_checkpoint"]:
                        self.info["info"] += " (plateau counter cleared)"
                        self._plateau_counter = 0
        return self._handle_trainer_terminate(last_score)


class mode_context(context_error_handler):
    """
    Help entering specific mode and recovering previous mode

    This is a context controller for entering specific mode at the beginning
    and back to previous mode at the end.

    Parameters
    ----------
    module : nn.Module, arbitrary PyTorch module.

    Examples
    --------
    >>> module = nn.Module()
    >>> with mode_context(module):
    >>>     pass  # do something

    """

    def __init__(
        self,
        module: nn.Module,
        *,
        to_train: Optional[bool],
        use_grad: Optional[bool],
    ):
        self._to_train = to_train
        self._module, self._training = module, module.training
        self._cache = {p: p.requires_grad for p in module.parameters()}
        if use_grad is not None:
            for p in module.parameters():
                p.requires_grad_(use_grad)
        if use_grad is None:
            self._grad_context: Optional[ContextManager] = None
        else:
            self._grad_context = torch.enable_grad() if use_grad else torch.no_grad()

    def __enter__(self) -> None:
        if self._to_train is not None:
            self._module.train(mode=self._to_train)
        if self._grad_context is not None:
            self._grad_context.__enter__()

    def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._to_train is not None:
            self._module.train(mode=self._training)
        if self._grad_context is not None:
            self._grad_context.__exit__(exc_type, exc_val, exc_tb)
        for p, v in self._cache.items():
            p.requires_grad_(v)


class train_context(mode_context):
    """
    Useful when we need to get gradients with our PyTorch model during evaluating.
    """

    def __init__(self, module: nn.Module, *, use_grad: bool = True):
        super().__init__(module, to_train=True, use_grad=use_grad)


class eval_context(mode_context):
    """
    Useful when we need to predict something with our PyTorch model during training.
    """

    def __init__(self, module: nn.Module, *, use_grad: bool = False):
        super().__init__(module, to_train=False, use_grad=use_grad)


class amp_autocast_context(context_error_handler):
    def __init__(self, use_amp: bool):
        if not use_amp:
            self._autocast = None
        else:
            if amp is None:
                raise ValueError("`amp` is not available but `use_amp` is set to True")
            self._autocast = amp.autocast()

    def __enter__(self) -> None:
        if self._autocast is not None:
            self._autocast.__enter__()

    def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._autocast is not None:
            self._autocast.__exit__(exc_type, exc_val, exc_tb)


__all__ = [
    "is_int",
    "is_float",
    "to_standard",
    "to_torch",
    "to_numpy",
    "to_2d",
    "to_prob",
    "collate_np_dicts",
    "collate_tensor_dicts",
    "switch_requires_grad",
    "get_gradient",
    "Initializer",
    "Activations",
    "TrainMonitor",
    "mode_context",
    "train_context",
    "eval_context",
    "amp_autocast_context",
]
