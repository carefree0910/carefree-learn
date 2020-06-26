import math
import torch
import logging

import numpy as np
import torch.nn as nn

from typing import *

from cftool.misc import *


def to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32))


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class Initializer(LoggingMixin):
    """
    Initializer for neural network weights

    Examples
    --------
    >>> initializer = Initializer({})
    >>> linear = nn.Linear(10, 10)
    >>> initializer.xavier(linear.weight)

    """

    defined_initialization = {"xavier", "normal", "truncated_normal"}
    custom_initializer = {}

    def __init__(self, config):
        self.config = config
        self._verbose_level = config.setdefault("verbose_level", 2)

    def initialize(self, param: nn.Parameter, method: str):
        custom_initializer = self.custom_initializer.get(method)
        if custom_initializer is None:
            return getattr(self, method)(param)
        return custom_initializer(self, param)

    @classmethod
    def add_initializer(cls, f, name):
        if name in cls.defined_initialization:
            print(f"{cls.warning_prefix}'{name}' initializer is already defined")
            return
        cls.defined_initialization.add(name)
        cls.custom_initializer[name] = f

    @staticmethod
    def xavier(param: nn.Parameter):
        nn.init.xavier_uniform_(param.data)

    def normal(self, param: nn.Parameter):
        mean = self.config.setdefault("mean", 0.)
        std = self.config.setdefault("std", 1.)
        with torch.no_grad():
            param.data.normal_(mean, std)

    def truncated_normal(self, param: nn.Parameter):
        span = self.config.setdefault("span", 2.)
        mean = self.config.setdefault("mean", 0.)
        std = self.config.setdefault("std", 1.)
        tol = self.config.setdefault("tol", 0.)
        epoch = self.config.setdefault("epoch", 20)
        n_elem = param.numel()
        weight_base = param.new_empty(n_elem).normal_()
        get_invalid = lambda w: (w > span) | (w < -span)
        invalid = get_invalid(weight_base)
        success = False
        for _ in range(epoch):
            n_invalid = int(invalid.sum())
            if n_invalid / n_elem <= tol:
                success = True
                break
            with torch.no_grad():
                weight_base[invalid] = param.new_empty(n_invalid).normal_()
                invalid = get_invalid(weight_base)
        if not success:
            self.log_msg(
                f"invalid ratio for truncated normal : {invalid.to(torch.float32).mean():8.6f}, "
                f"it might cause by too little epoch ({epoch}) or too small tolerance ({tol})",
                self.warning_prefix, 2, logging.WARNING
            )
        with torch.no_grad():
            param.data.copy_(weight_base.reshape(param.shape))
            param.data.mul_(std).add_(mean)


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

    def __init__(self,
                 configs: Dict[str, Any] = None):
        if configs is None:
            configs = {}
        self.configs = configs

    def __getattr__(self, item):
        try:
            return getattr(nn, item)(**self.configs.setdefault(item, {}))
        except AttributeError:
            raise NotImplementedError(
                f"neither pytorch nor custom Activations implemented activation '{item}'")

    def module(self,
               name: str) -> nn.Module:
        if name is None:
            return nn.Identity()
        return getattr(self, name)

    # publications

    @property
    def mish(self):

        class Mish(nn.Module):
            def forward(self, x):
                return x * (torch.tanh(nn.functional.softplus(x)))

        return Mish()

    # custom

    @property
    def sign(self):

        class Sign(nn.Module):
            def forward(self, x):
                return torch.sign(x)

        return Sign()

    @property
    def one_hot(self):

        class OneHot(nn.Module):
            def forward(self, x):
                return x * (x == torch.max(x, dim=1, keepdim=True)[0]).to(torch.float32)

        return OneHot()

    @property
    def multiplied_tanh(self):

        class MultipliedTanh(nn.Tanh):
            def __init__(self, ratio, trainable=True):
                super().__init__()
                ratio = torch.tensor([ratio], dtype=torch.float32)
                self.ratio = ratio if not trainable else nn.Parameter(ratio)

            def forward(self, x):
                x = x * self.ratio
                return super().forward(x)

        return MultipliedTanh(**self.configs.setdefault("multiplied_tanh", {}))

    @property
    def multiplied_softmax(self):

        class MultipliedSoftmax(nn.Softmax):
            def __init__(self, ratio, dim=1, trainable=True):
                super().__init__(dim)
                ratio = torch.tensor([ratio], dtype=torch.float32)
                self.ratio = ratio if not trainable else nn.Parameter(ratio)

            def forward(self, x):
                x = x * self.ratio
                return super().forward(x)

        return MultipliedSoftmax(**self.configs.setdefault("multiplied_softmax", {}))


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
    * `register_pipeline` method MUST be called before monitoring
    * instance passed to`register_pipeline` method MUST be a subclass of `LoggingMixin`,
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
    >>> monitor = TrainMonitor(metric.sign).register_pipeline(model)
    >>> n_epoch, epoch_count = 20, 0
    >>> while epoch_count <= n_epoch:
    >>>     model.train()
    >>>     predictions = model.predict(x)
    >>>     if monitor.check_terminate(metric.metric(y, predictions)):
    >>>         break

    """

    def __init__(self,
                 sign,
                 num_scores_per_snapshot=1,
                 history_ratio=3,
                 tolerance_ratio=2,
                 extension=5,
                 std_floor=0.001,
                 std_ceiling=0.01,
                 aggressive=False):
        self.sign = sign
        self.num_scores_per_snapshot = num_scores_per_snapshot
        self.num_history = int(num_scores_per_snapshot * history_ratio)
        self.num_tolerance = int(num_scores_per_snapshot * tolerance_ratio)
        self.extension = extension
        self.is_aggressive = aggressive
        self.std_floor, self.std_ceiling = std_floor, std_ceiling
        self._scores = []
        self.plateau_flag = False
        self._is_best = self._running_best = None
        self._descend_increment = self.num_history * extension / 30
        self._incrementer = Incrementer(self.num_history)

        self._over_fit_performance = math.inf
        self._best_checkpoint_performance = -math.inf
        self._descend_counter = self._plateau_counter = self.over_fitting_flag = 0
        self.info = {"terminate": False, "save_checkpoint": False, "save_best": aggressive, "info": None}

    @property
    def log_msg(self):
        return self._pipeline.log_msg

    @property
    def plateau_threshold(self):
        return 6 * self.num_tolerance * self.num_history

    def _update_running_info(self, last_score):
        self._incrementer.update(last_score)
        if self._running_best is None:
            if self._scores[0] > self._scores[1]:
                improvement = 0
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

    def _handle_overfitting(self, last_score, res, std):
        if self._descend_counter == 0:
            self.info["save_best"] = True
            self._over_fit_performance = last_score
        self._descend_counter += min(self.num_tolerance / 3, -res / std)
        self.log_msg(
            f"descend counter updated : {self._descend_counter:6.4f}",
            self._pipeline.info_prefix, 6, logging.DEBUG
        )
        self.over_fitting_flag = 1

    def _handle_recovering(self, improvement, last_score, res, std):
        if res > 3 * std and self._is_best and improvement > std:
            self.info["save_best"] = True
        new_counter = self._descend_counter - res / std
        if self._descend_counter > 0 >= new_counter:
            self._over_fit_performance = math.inf
            if last_score > self._best_checkpoint_performance:
                self._best_checkpoint_performance = last_score
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
                self._pipeline.info_prefix, 6, logging.DEBUG
            )

    def _handle_is_best(self):
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

    def _handle_period(self, last_score):
        if self.is_aggressive:
            return
        if len(self._scores) % self.num_scores_per_snapshot == 0 and \
                last_score > self._best_checkpoint_performance:
            self._best_checkpoint_performance = last_score
            self._plateau_counter //= 2
            self.info["terminate"] = False
            self.info["save_checkpoint"] = True
            self.info["info"] = (
                f"current snapshot ({len(self._scores)}) leads to best checkpoint we've ever had, "
                "saving checkpoint in case we need to restore"
            )

    def _punish_extension(self):
        self.plateau_flag = True
        self._descend_counter += self._descend_increment

    def _handle_pipeline_terminate(self):
        pipeline = self._pipeline
        if self.info["terminate"]:
            self.log_msg(
                f"early stopped at n_epoch={pipeline._epoch_count} due to '{self.info['info']}'",
                prefix=pipeline.info_prefix
            )
            return True
        if self.info["save_checkpoint"]:
            self.log_msg(f"{self.info['info']}", pipeline.info_prefix, 3)
            pipeline.save_checkpoint()
        if (
            pipeline._epoch_count == pipeline.num_epoch and
            pipeline._epoch_count < pipeline.max_epoch and
            not self.info["terminate"]
        ):
            self._punish_extension()
            new_epoch = pipeline.num_epoch + self.extension
            pipeline.num_epoch = min(new_epoch, pipeline.max_epoch)
            self.log_msg(f"extending num_epoch to {pipeline.num_epoch}", pipeline.info_prefix, 3)
        if pipeline._epoch_count == pipeline.max_epoch:
            if not self.info["terminate"]:
                self.log_msg(
                    "model seems to be under-fitting but max_epoch reached, "
                    "increasing max_epoch may improve performance.", pipeline.info_prefix
                )
            return True
        return False

    def register_pipeline(self, pipeline):
        self._pipeline = pipeline
        return self

    def check_terminate(self, new_metric):
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
                    self._plateau_counter += self.std_floor / max(std, self.std_floor / 6)
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
                    f"plateau counter updated : {self._plateau_counter:>6.4f} / {self.plateau_threshold}",
                    self._pipeline.info_prefix, 6, logging.DEBUG
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
        return self._handle_pipeline_terminate()


class eval_context(context_error_handler):
    """
    Help entering eval mode and recovering previous mode

    This is a context controller for entering eval mode at the beginning
    and back to previous mode at the end.

    Useful when we need to predict something with our PyTorch model during training.

    Parameters
    ----------
    module : nn.Module, arbitrary PyTorch module.

    Examples
    --------
    >>> module = nn.Module()
    >>> with eval_context(module):
    >>>     pass  # do something

    """

    def __init__(self,
                 module: nn.Module,
                 *,
                 no_grad: bool = True):
        self._module, self._training = module, module.training
        self._params_required_grad = [param for param in module.parameters() if param.requires_grad]
        tuple(map(lambda param: param.requires_grad_(False), self._params_required_grad))
        self._no_grad = torch.no_grad() if no_grad else None

    def __enter__(self):
        self._module.eval()
        if self._no_grad is not None:
            self._no_grad.__enter__()

    def _normal_exit(self, exc_type, exc_val, exc_tb):
        self._module.train(mode=self._training)
        if self._no_grad is not None:
            self._no_grad.__exit__(exc_type, exc_val, exc_tb)
        tuple(map(lambda param: param.requires_grad_(True), self._params_required_grad))


__all__ = [
    "to_torch", "to_numpy",
    "Initializer", "Activations", "TrainMonitor", "eval_context"
]
