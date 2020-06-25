import os

from typing import *
from cftool.misc import *
from cftool.ml.utils import *
from cftool.ml.param_utils import *
from cfdata.tabular import *
from functools import partial
from cftool.ml.hpo import HPOBase

from .dist import *
from .bases import *
from .models import *
from .modules import *
from .misc.toolkit import eval_context, Initializer


# register

def register_initializer(name):
    def _register(f):
        Initializer.add_initializer(f, name)
        return f
    return _register


# API

def make(model: str = "fcnn",
         *,
         delim: str = None,
         skip_first: bool = None,
         cv_ratio: float = 0.1,
         min_epoch: int = None,
         num_epoch: int = None,
         max_epoch: int = None,
         batch_size: int = None,
         logging_path: str = None,
         data_config: Dict[str, Any] = None,
         read_config: Dict[str, Any] = None,
         model_config: Dict[str, Any] = None,
         metrics: Union[str, List[str]] = None,
         metric_config: Dict[str, Any] = None,
         optimizer: str = None,
         optimizer_config: Dict[str, Any] = None,
         optimizers: Dict[str, Any] = None,
         trigger_logging: bool = None,
         cuda: Union[int, str] = 0,
         verbose_level: int = 2,
         use_tqdm: bool = True,
         **kwargs) -> Wrapper:
    # wrapper general
    kwargs["model"] = model
    kwargs["cv_ratio"] = cv_ratio
    if data_config is not None:
        kwargs["data_config"] = data_config
    if read_config is None:
        read_config = {}
    if delim is not None:
        read_config["delim"] = delim
    if skip_first is not None:
        read_config["skip_first"] = skip_first
    kwargs["read_config"] = read_config
    if model_config is not None:
        kwargs["model_config"] = model_config
    if logging_path is not None:
        kwargs["logging_path"] = logging_path
    if trigger_logging is not None:
        kwargs["trigger_logging"] = trigger_logging
    # pipeline general
    pipeline_config = kwargs.setdefault("pipeline_config", {})
    pipeline_config["use_tqdm"] = use_tqdm
    if min_epoch is not None:
        pipeline_config["min_epoch"] = min_epoch
    if num_epoch is not None:
        pipeline_config["num_epoch"] = num_epoch
    if max_epoch is not None:
        pipeline_config["max_epoch"] = max_epoch
    if batch_size is not None:
        pipeline_config["batch_size"] = batch_size
    # metrics
    if metric_config is not None:
        if metrics is not None:
            print(
                f"{LoggingMixin.warning_prefix}`metrics` is set to '{metrics}' "
                f"but `metric_config` is provided, so `metrics` will be ignored")
    elif metrics is not None:
        metric_config = {"types": metrics}
    if metric_config is not None:
        pipeline_config["metric_config"] = metric_config
    # optimizers
    if optimizers is not None:
        if optimizer is not None:
            print(
                f"{LoggingMixin.warning_prefix}`optimizer` is set to '{optimizer}' "
                f"but `optimizers` is provided, so `optimizer` will be ignored")
        if optimizer_config is not None:
            print(
                f"{LoggingMixin.warning_prefix}`optimizer_config` is set to '{optimizer_config}' "
                f"but `optimizers` is provided, so `optimizer_config` will be ignored")
    elif optimizer is not None:
        if optimizer_config is None:
            optimizer_config = {}
        optimizers = {"all": {"optimizer": optimizer, "optimizer_config": optimizer_config}}
    if optimizers is not None:
        pipeline_config["optimizers"] = optimizers
    return Wrapper(kwargs, cuda=cuda, verbose_level=verbose_level)
