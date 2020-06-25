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


SAVING_DELIM = "^_^"
wrappers_dict_type = Dict[str, Wrapper]
wrappers_type = Union[Wrapper, List[Wrapper], wrappers_dict_type]
repeat_result_type = Tuple[EvaluateTransformer, Union[List[ModelPattern], Dict[str, List[ModelPattern]]]]


def _to_saving_path(identifier: str,
                    saving_folder: str) -> str:
    if saving_folder is None:
        saving_path = identifier
    else:
        saving_path = os.path.join(saving_folder, identifier)
    return saving_path


def _make_saving_path(name: str,
                      saving_path: str,
                      remove_existing: bool) -> str:
    saving_path = os.path.abspath(saving_path)
    saving_folder, identifier = os.path.split(saving_path)
    postfix = f"{SAVING_DELIM}{name}"
    if os.path.isdir(saving_folder) and remove_existing:
        for existing_model in os.listdir(saving_folder):
            if os.path.isdir(os.path.join(saving_folder, existing_model)):
                continue
            if existing_model.startswith(f"{identifier}{postfix}"):
                print(f"{LoggingMixin.warning_prefix}"
                      f"'{existing_model}' was found, it will be removed")
                os.remove(os.path.join(saving_folder, existing_model))
    return f"{saving_path}{postfix}"


def _to_wrappers(wrappers: wrappers_type) -> wrappers_dict_type:
    if not isinstance(wrappers, dict):
        if not isinstance(wrappers, list):
            wrappers = [wrappers]
        names = [wrapper.model.__identifier__ for wrapper in wrappers]
        if len(set(names)) != len(wrappers):
            raise ValueError("wrapper names are not provided but identical wrapper.model is detected")
        wrappers = dict(zip(names, wrappers))
    return wrappers


def save(wrappers: wrappers_type,
         identifier: str = "cflearn",
         saving_folder: str = None) -> wrappers_dict_type:
    wrappers = _to_wrappers(wrappers)
    saving_path = _to_saving_path(identifier, saving_folder)
    for name, wrapper in wrappers.items():
        wrapper.save(_make_saving_path(name, saving_path, True), compress=True)
    return wrappers


def load(identifier: str = "cflearn",
         saving_folder: str = None) -> wrappers_dict_type:
    wrappers = {}
    saving_path = _to_saving_path(identifier, saving_folder)
    saving_path = os.path.abspath(saving_path)
    base_folder = os.path.dirname(saving_path)
    for existing_model in os.listdir(base_folder):
        if not os.path.isfile(os.path.join(base_folder, existing_model)):
            continue
        existing_model, existing_extension = os.path.splitext(existing_model)
        if existing_extension != ".zip":
            continue
        if SAVING_DELIM in existing_model:
            *folder, name = existing_model.split(SAVING_DELIM)
            if os.path.join(base_folder, SAVING_DELIM.join(folder)) != saving_path:
                continue
            wrappers[name] = Wrapper.load(_make_saving_path(name, saving_path, False), compress=True)
    if not wrappers:
        raise ValueError(f"'{saving_path}' was not a valid saving path")
    return wrappers
