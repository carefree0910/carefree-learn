import os
import json
import math
import time
import torch
import shutil
import inspect
import logging
import torchvision

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from PIL import ImageDraw
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Generic
from typing import TypeVar
from typing import Callable
from typing import Optional
from typing import ContextManager
from argparse import Namespace
from datetime import datetime
from datetime import timedelta
from collections import defaultdict
from collections import OrderedDict
from cftool.misc import prod
from cftool.misc import hash_code
from cftool.misc import register_core
from cftool.misc import show_or_save
from cftool.misc import shallow_copy_dict
from cftool.misc import context_error_handler
from cftool.misc import LoggingMixin

from ..types import arr_type
from ..types import data_type
from ..types import param_type
from ..types import tensor_dict_type
from ..types import general_config_type
from ..types import sample_weights_type
from ..constants import INPUT_KEY
from ..constants import TIME_FORMAT
from ..constants import WARNING_PREFIX

try:
    import SharedArray as sa
except:
    sa = None


# general


def _parse_config(config: general_config_type) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, str):
        with open(config, "r") as f:
            return json.load(f)
    return shallow_copy_dict(config)


def prepare_workplace_from(workplace: str, timeout: timedelta = timedelta(7)) -> str:
    current_time = datetime.now()
    if os.path.isdir(workplace):
        for stuff in os.listdir(workplace):
            if not os.path.isdir(os.path.join(workplace, stuff)):
                continue
            try:
                stuff_time = datetime.strptime(stuff, TIME_FORMAT)
                stuff_delta = current_time - stuff_time
                if stuff_delta > timeout:
                    print(
                        f"{WARNING_PREFIX}{stuff} will be removed "
                        f"(already {stuff_delta} ago)"
                    )
                    shutil.rmtree(os.path.join(workplace, stuff))
            except:
                pass
    workplace = os.path.join(workplace, current_time.strftime(TIME_FORMAT))
    os.makedirs(workplace)
    return workplace


def get_latest_workplace(root: str) -> Optional[str]:
    all_workplaces = []
    for stuff in os.listdir(root):
        if not os.path.isdir(os.path.join(root, stuff)):
            continue
        try:
            datetime.strptime(stuff, TIME_FORMAT)
            all_workplaces.append(stuff)
        except:
            pass
    if not all_workplaces:
        return None
    return os.path.join(root, sorted(all_workplaces)[-1])


def sort_dict_by_value(d: Dict[Any, Any], *, reverse: bool = False) -> OrderedDict:
    sorted_items = sorted([(v, k) for k, v in d.items()], reverse=reverse)
    return OrderedDict({item[1]: item[0] for item in sorted_items})


def to_standard(arr: np.ndarray) -> np.ndarray:
    if is_int(arr):
        arr = arr.astype(np.int64)
    elif is_float(arr):
        arr = arr.astype(np.float32)
    return arr


def parse_args(args: Any) -> Namespace:
    return Namespace(**{k: None if not v else v for k, v in args.__dict__.items()})


def parse_path(path: Optional[str], root_dir: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if root_dir is None:
        return path
    return os.path.abspath(os.path.join(root_dir, path))


def get_arguments(*, pop_class_attributes: bool = True) -> Dict[str, Any]:
    frame = inspect.currentframe().f_back  # type: ignore
    if frame is None:
        raise ValueError("`get_arguments` should be called inside a frame")
    arguments = inspect.getargvalues(frame)[-1]
    if pop_class_attributes:
        arguments.pop("self", None)
        arguments.pop("__class__", None)
    return arguments


def _rmtree(folder: str, patience: float = 10.0) -> None:
    if not os.path.isdir(folder):
        return None
    t = time.time()
    while True:
        try:
            if time.time() - t >= patience:
                prefix = LoggingMixin.warning_prefix
                print(f"\n{prefix}failed to rmtree: {folder}")
                break
            shutil.rmtree(folder)
            break
        except:
            print("", end=".", flush=True)
            time.sleep(1)


T = TypeVar("T")


class WithRegister(Generic[T]):
    d: Dict[str, Type[T]]
    __identifier__: str

    @classmethod
    def get(cls, name: str) -> Type[T]:
        return cls.d[name]

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> T:
        return cls.get(name)(**config)  # type: ignore

    @classmethod
    def make_multiple(
        cls,
        names: Union[str, List[str]],
        configs: Optional[Dict[str, Any]] = None,
    ) -> Union[T, List[T]]:
        if configs is None:
            configs = {}
        if isinstance(names, str):
            return cls.make(names, configs)  # type: ignore
        return [
            cls.make(name, shallow_copy_dict(configs.get(name, {})))  # type: ignore
            for name in names
        ]

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, cls.d, before_register=before)


class WeightsStrategy:
    def __init__(self, strategy: Optional[str]):
        self.strategy = strategy

    def __call__(self, num_train: int, num_valid: int) -> sample_weights_type:
        if self.strategy is None:
            return None
        return getattr(self, self.strategy)(num_train, num_valid)

    def linear_decay(self, num_train: int, num_valid: int) -> sample_weights_type:
        return np.linspace(0, 1, num_train + 1)[1:]

    def radius_decay(self, num_train: int, num_valid: int) -> sample_weights_type:
        return np.sin(np.arccos(1.0 - np.linspace(0, 1, num_train + 1)[1:]))

    def log_decay(self, num_train: int, num_valid: int) -> sample_weights_type:
        return np.log(np.arange(num_train) + np.e)

    def sigmoid_decay(self, num_train: int, num_valid: int) -> sample_weights_type:
        x = np.linspace(-5.0, 5.0, num_train)
        return 1.0 / (1.0 + np.exp(-x))

    def visualize(self, export_path: str = "weights_strategy.png") -> None:
        n = 1000
        x = np.linspace(0, 1, n)
        y = self(n, 0)
        if isinstance(y, tuple):
            y = y[0]
        plt.figure()
        plt.plot(x, y)
        show_or_save(export_path)


class LoggingMixinWithRank(LoggingMixin):
    is_rank_0: bool = True

    def set_rank_0(self, value: bool) -> None:
        self.is_rank_0 = value
        for v in self.__dict__.values():
            if isinstance(v, LoggingMixinWithRank):
                v.set_rank_0(value)

    def _init_logging(
        self,
        verbose_level: Optional[int] = 2,
        trigger: bool = True,
    ) -> None:
        if not self.is_rank_0:
            return None
        super()._init_logging(verbose_level, trigger)

    def log_msg(
        self,
        body: str,
        prefix: str = "",
        verbose_level: Optional[int] = 1,
        msg_level: int = logging.INFO,
        frame: Any = None,
    ) -> None:
        if not self.is_rank_0:
            return None
        super().log_msg(body, prefix, verbose_level, msg_level, frame)

    def log_block_msg(
        self,
        body: str,
        prefix: str = "",
        title: str = "",
        verbose_level: Optional[int] = 1,
        msg_level: int = logging.INFO,
        frame: Any = None,
    ) -> None:
        if not self.is_rank_0:
            return None
        super().log_block_msg(body, prefix, title, verbose_level, msg_level, frame)

    def log_timing(self) -> None:
        if not self.is_rank_0:
            return None
        return super().log_timing()


# dl


def to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(to_standard(arr))


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def to_device(batch: tensor_dict_type, device: torch.device) -> tensor_dict_type:
    return {
        k: v if not isinstance(v, torch.Tensor) else v.to(device)
        for k, v in batch.items()
    }


def softmax(arr: arr_type) -> arr_type:
    if isinstance(arr, torch.Tensor):
        return F.softmax(arr, dim=1)
    logits = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(1, keepdims=True)


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


def set_requires_grad(module: nn.Module, requires_grad: bool = False) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def scheduler_requires_metric(scheduler: Any) -> bool:
    signature = inspect.signature(scheduler.step)
    for name, param in signature.parameters.items():
        if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if name == "metrics":
                return True
    return False


# This is a modified version of https://github.com/sksq96/pytorch-summary
#  So it can summary `carefree-learn` model structures better
def summary(
    model: nn.Module,
    sample_batch: tensor_dict_type,
    *,
    return_only: bool = False,
) -> str:
    def _get_param_counts(module_: nn.Module) -> Tuple[int, int]:
        num_params = 0
        num_trainable_params = 0
        for param in module_.parameters():
            local_num_params = int(round(prod(param.data.shape)))
            num_params += local_num_params
            if param.requires_grad:
                num_trainable_params += local_num_params
        return num_params, num_trainable_params

    def register_hook(module: nn.Module) -> None:
        def hook(module_: nn.Module, inp: Any, output: Any) -> None:
            m_name = module_names.get(module_)
            if m_name is None:
                return

            inp = inp[0]
            if not isinstance(inp, torch.Tensor):
                return
            if isinstance(output, (list, tuple)):
                for element in output:
                    if not isinstance(element, torch.Tensor):
                        return
            elif not isinstance(output, torch.Tensor):
                return

            m_dict: OrderedDict[str, Any] = OrderedDict()
            m_dict["input_shape"] = list(inp.size())
            if len(m_dict["input_shape"]) > 0:
                m_dict["input_shape"][0] = -1
            if isinstance(output, (list, tuple)):
                m_dict["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
                m_dict["is_multiple_output"] = True
            else:
                m_dict["output_shape"] = list(output.size())
                if len(m_dict["output_shape"]) > 0:
                    m_dict["output_shape"][0] = -1
                m_dict["is_multiple_output"] = False

            num_params_, num_trainable_params_ = _get_param_counts(module_)
            m_dict["num_params"] = num_params_
            m_dict["num_trainable_params"] = num_trainable_params_
            raw_summary_dict[m_name] = m_dict

        if not isinstance(module, torch.jit.ScriptModule):
            hooks.append(module.register_forward_hook(hook))

    # get names
    def _inject_names(m: nn.Module, previous_names: List[str]) -> None:
        info_list = []
        for child in m.children():
            current_names = previous_names + [type(child).__name__]
            current_name = ".".join(current_names)
            module_names[child] = current_name
            info_list.append((child, current_name, current_names))
        counts: Dict[str, int] = defaultdict(int)
        idx_mapping: Dict[nn.Module, int] = {}
        for child, current_name, _ in info_list:
            idx_mapping[child] = counts[current_name]
            counts[current_name] += 1
        for child, current_name, current_names in info_list:
            if counts[current_name] == 1:
                continue
            current_name = f"{current_name}-{idx_mapping[child]}"
            module_names[child] = current_name
            current_names[-1] = current_name.split(".")[-1]
        for child, _, current_names in info_list:
            _inject_names(child, current_names)

    module_names: OrderedDict[nn.Module, str] = OrderedDict()
    existing_names: Set[str] = set()

    def _get_name(original: str) -> str:
        count = 0
        final_name = original
        while final_name in existing_names:
            count += 1
            final_name = f"{original}_{count}"
        existing_names.add(final_name)
        return final_name

    model_name = _get_name(type(model).__name__)
    module_names[model] = model_name
    _inject_names(model, [model_name])

    # create properties
    raw_summary_dict: OrderedDict[str, Any] = OrderedDict()
    hooks: List[Any] = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    with eval_context(model):
        if not hasattr(model, "summary_forward"):
            model(0, sample_batch)
        else:
            model.summary_forward(0, sample_batch)  # type: ignore

    # remove these hooks
    for h in hooks:
        h.remove()

    # get hierarchy
    hierarchy: OrderedDict[str, Any] = OrderedDict()
    for key in raw_summary_dict:
        split = key.split(".")
        d = hierarchy
        for elem in split[:-1]:
            d = d.setdefault(elem, OrderedDict())
        d.setdefault(split[-1], None)

    # reconstruct summary_dict
    def _inject_summary(current_hierarchy: Any, previous_keys: List[str]) -> None:
        if previous_keys and not previous_keys[-1]:
            previous_keys.pop()
        current_layer = len(previous_keys)
        current_count = hierarchy_counts.get(current_layer, 0)
        prefix = "  " * current_layer
        for k, v in current_hierarchy.items():
            current_keys = previous_keys + [k]
            concat_k = ".".join(current_keys)
            current_summary = raw_summary_dict.get(concat_k)
            summary_dict[f"{prefix}{k}-{current_count}"] = current_summary
            hierarchy_counts[current_layer] = current_count + 1
            if v is not None:
                _inject_summary(v, current_keys)

    hierarchy_counts: Dict[int, int] = {}
    summary_dict: OrderedDict[str, Any] = OrderedDict()
    _inject_summary(hierarchy, [])

    line_length = 120
    messages = ["=" * line_length]
    line_format = "{:30}  {:>20} {:>40} {:>20}"
    headers = "Layer (type)", "Input Shape", "Output Shape", "Trainable Param #"
    line_new = line_format.format(*headers)
    messages.append(line_new)
    messages.append("-" * line_length)
    total_output = 0
    for layer, layer_summary in summary_dict.items():
        layer_name = "-".join(layer.split("-")[:-1])
        if layer_summary is None:
            line_new = line_format.format(layer_name, "", "", "")
        else:
            line_new = line_format.format(
                layer_name,
                str(layer_summary["input_shape"]),
                str(layer_summary["output_shape"]),
                "{0:,}".format(layer_summary["num_trainable_params"]),
            )
            output_shape = layer_summary["output_shape"]
            is_multiple_output = layer_summary["is_multiple_output"]
            if not is_multiple_output:
                output_shape = [output_shape]
            for shape in output_shape:
                total_output += prod(shape)
        messages.append(line_new)

    total_params, trainable_params = _get_param_counts(model)
    # assume 4 bytes/number (float on cuda).
    x_batch = sample_batch[INPUT_KEY]
    total_input_size = abs(prod(x_batch.shape[1:]) * 4.0 / (1024 ** 2.0))
    # x2 for gradients
    total_output_size = abs(2.0 * total_output * 4.0 / (1024 ** 2.0))
    total_params_size = abs(total_params * 4.0 / (1024 ** 2.0))
    total_size = total_params_size + total_output_size + total_input_size

    non_trainable_params = total_params - trainable_params
    messages.append("=" * line_length)
    messages.append("Total params: {0:,}".format(total_params))
    messages.append("Trainable params: {0:,}".format(trainable_params))
    messages.append("Non-trainable params: {0:,}".format(non_trainable_params))
    messages.append("-" * line_length)
    messages.append("Input size (MB): %0.2f" % total_input_size)
    messages.append("Forward/backward pass size (MB): %0.2f" % total_output_size)
    messages.append("Params size (MB): %0.2f" % total_params_size)
    messages.append("Estimated Total Size (MB): %0.2f" % total_size)
    messages.append("-" * line_length)
    msg = "\n".join(messages)
    if not return_only:
        print(msg)
    return msg


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

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
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


class Initializer(LoggingMixinWithRank):
    """
    Initializer for neural network weights

    Examples
    --------
    >>> initializer = Initializer()
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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._verbose_level = self.config.setdefault("verbose_level", 2)

    def initialize(self, param: param_type, method: str) -> Any:
        custom_initializer = self.custom_initializer.get(method)
        if custom_initializer is None:
            return getattr(self, method)(param)
        return custom_initializer(self, param)

    @classmethod
    def add_initializer(cls, f: Callable, name: str) -> None:
        if name in cls.defined_initialization:
            print(f"{WARNING_PREFIX}'{name}' initializer is already defined")
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

    def orthogonal(self, param: param_type) -> None:
        gain = self.config.setdefault("gain", 1.0)
        nn.init.orthogonal_(param.data, gain)


# ml


def is_int(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.integer)


def is_float(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.floating)


def to_2d(arr: data_type) -> data_type:
    if arr is None or isinstance(arr, str):
        return None
    if isinstance(arr, np.ndarray):
        return arr.reshape([len(arr), -1])
    if isinstance(arr[0], list):
        return arr
    return [[elem] for elem in arr]  # type: ignore


def corr(
    predictions: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    get_diagonal: bool = False,
) -> torch.Tensor:
    w_sum = 0.0 if weights is None else weights.sum().item()
    if weights is None:
        mean = predictions.mean(0, keepdim=True)
    else:
        mean = (predictions * weights).sum(0, keepdim=True) / w_sum
    vp = predictions - mean
    if weights is None:
        vp_norm = torch.norm(vp, 2, dim=0, keepdim=True)
    else:
        vp_norm = (weights * (vp ** 2)).sum(0, keepdim=True).sqrt()
    if predictions is target:
        mat = vp.t().matmul(vp) / (vp_norm * vp_norm.t())
    else:
        if weights is None:
            target_mean = target.mean(0, keepdim=True)
        else:
            target_mean = (target * weights).sum(0, keepdim=True) / w_sum
        vt = (target - target_mean).t()
        if weights is None:
            vt_norm = torch.norm(vt, 2, dim=1, keepdim=True)
        else:
            vt_norm = (weights.t() * (vt ** 2)).sum(1, keepdim=True).sqrt()
        mat = vt.matmul(vp) / (vp_norm * vt_norm)
    if not get_diagonal:
        return mat
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(
            "`get_diagonal` is set to True but the correlation matrix "
            "is not a squared matrix, which is an invalid condition"
        )
    return mat.diag()


def _to_address(path: str) -> str:
    return f"shared_{hash_code(path)}"


def _check_sa(address: str) -> bool:
    if sa is None:
        return False
    try:
        sa.attach(address)
        return True
    except FileNotFoundError:
        return False


def _write_shared(address: str, array: np.ndarray) -> None:
    try:
        shared = sa.attach(address)
    except FileNotFoundError:
        shared = sa.create(address, array.shape, array.dtype)
    shared[:] = array


class SharedArrayWrapper:
    def __init__(self, root: str, path: str, *, to_memory: bool):
        self.path = os.path.join(root, path)
        self.folder, file = os.path.split(self.path)
        os.makedirs(self.folder, exist_ok=True)
        self.flag_path = os.path.join(self.folder, f"flag_of_{file}")
        self.address, self.flag_address = map(_to_address, [self.path, self.flag_path])
        if to_memory and sa is None:
            print(
                f"{WARNING_PREFIX}`to_memory` is set to True but `SharedArray` lib "
                f"is not available, therefore `to_memory` will be set to False"
            )
            to_memory = False
        self.to_memory = to_memory

    @property
    def is_ready(self) -> bool:
        if self.to_memory:
            if not _check_sa(self.address):
                return False
            if not _check_sa(self.flag_address):
                return False
            return sa.attach(self.flag_address).item()
        if not os.path.isfile(self.path):
            return False
        if not os.path.isfile(self.flag_path):
            return False
        return bool(np.load(self.flag_path, mmap_mode="r").item())

    def read(self, *, writable: bool = False) -> np.ndarray:
        if self.to_memory:
            arr = sa.attach(self.address)
            arr.flags.writeable = writable
            return arr
        return np.load(self.path, mmap_mode="r+" if writable else "r")

    def write(self, arr: np.ndarray) -> None:
        self._write(arr, overwrite=True, is_finished=True)

    def prepare(self, arr: np.ndarray) -> None:
        # prepare an empty array at certain path / address
        # this is mainly for multiprocessing
        self._write(arr, overwrite=False, is_finished=False)

    def mark_finished(self) -> None:
        flag = self.read(writable=True)
        flag[0] = True

    def delete(self) -> None:
        if self.is_ready:
            print(f"> removing {self.path} & {self.flag_path}")
            if self.to_memory:
                sa.delete(self.address)
                sa.delete(self.flag_address)
                return None
            os.remove(self.path)
            os.remove(self.flag_path)

    def _give_permission(self) -> None:
        os.system(f"chmod -R 777 {self.folder}")

    def _write(
        self,
        arr: np.ndarray,
        *,
        is_finished: bool,
        overwrite: bool = True,
    ) -> None:
        if self.is_ready and overwrite:
            path = self.address if self.to_memory else self.path
            print(f"> there's already an array at '{path}', " "it will be overwritten")
            self.delete()
        if self.to_memory:
            _write_shared(self.address, arr)
            _write_shared(self.flag_address, np.array([is_finished]))
            return None
        np.save(self.path, arr)
        np.save(self.flag_path, np.array([is_finished]))
        self._give_permission()


# cv


def interpolate(
    src: torch.Tensor,
    *,
    mode: str = "nearest",
    factor: Optional[float] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    anchor: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    if "linear" in mode or mode == "bicubic":
        kwargs.setdefault("align_corners", False)
    if factor is None:
        if size is None:
            if anchor is None:
                raise ValueError("either `size` or `anchor` should be provided")
            size = (anchor.shape[2], anchor.shape[3])
        if isinstance(size, int):
            if src.shape[2] == src.shape[3] == size:
                return src
        elif src.shape[2] == size[0] and src.shape[3] == size[1]:
            return src
        return F.interpolate(src, size=size, mode=mode, **kwargs)
    template = "`{}` will take no affect because `factor` is provided"
    if size is not None:
        print(f"{WARNING_PREFIX}{template.format('size')}")
    if anchor is not None:
        print(f"{WARNING_PREFIX}{template.format('anchor')}")
    if factor == 1.0:
        return src
    return F.interpolate(
        src,
        mode=mode,
        scale_factor=factor,
        recompute_scale_factor=True,
        **kwargs,
    )


def save_images(arr: arr_type, path: str, n_row: Optional[int] = None) -> None:
    if isinstance(arr, np.ndarray):
        arr = to_torch(arr)
    if n_row is None:
        n_row = math.ceil(math.sqrt(len(arr)))
    torchvision.utils.save_image(arr, path, normalize=True, nrow=n_row)


def iou(logits: arr_type, labels: arr_type) -> arr_type:
    is_torch = isinstance(logits, torch.Tensor)
    num_classes = logits.shape[1]
    if num_classes == 1:
        if is_torch:
            heat_map = torch.sigmoid(logits)
        else:
            heat_map = 1.0 / (1.0 + np.exp(-logits))
    elif num_classes == 2:
        heat_map = softmax(logits)[:, [1]]
    else:
        raise ValueError("`IOU` only supports binary situations")
    intersect = heat_map * labels
    union = heat_map + labels - intersect
    kwargs = {"dim" if is_torch else "axis": tuple(range(1, len(intersect.shape)))}
    return intersect.sum(**kwargs) / union.sum(**kwargs)


def is_gray(arr: arr_type) -> bool:
    if isinstance(arr, np.ndarray):
        return arr.shape[-1] == 1
    if len(arr.shape) == 3:
        return arr.shape[0] == 1
    return arr.shape[1] == 1


def min_max_normalize(arr: arr_type, *, global_norm: bool = True) -> arr_type:
    eps = 1.0e-8
    if global_norm:
        arr_min, arr_max = arr.min().item(), arr.max().item()
        return (arr - arr_min) / max(eps, arr_max - arr_min)
    if isinstance(arr, np.ndarray):
        arr_min, arr_max = arr.min(axis=0), arr.max(axis=0)
        diff = np.maximum(eps, arr_max - arr_min)
    else:
        arr_min, arr_max = arr.min(dim=0).values, arr.max(dim=0).values
        diff = torch.clip(arr_max - arr_min, max=eps)
    return (arr - arr_min) / diff


def imagenet_normalize(arr: arr_type) -> arr_type:
    mean_gray, std_gray = [0.485], [0.229]
    mean_rgb, std_rgb = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    np_constructor = lambda inp: np.array(inp, dtype=np.float32).reshape([1, 1, -1])
    torch_constructor = lambda inp: torch.tensor(inp, device=arr.device).view(-1, 1, 1)
    constructor = np_constructor if isinstance(arr, np.ndarray) else torch_constructor
    if is_gray(arr):
        mean, std = map(constructor, [mean_gray, std_gray])
    else:
        mean, std = map(constructor, [mean_rgb, std_rgb])
    return (arr - mean) / std


def make_indices_visualization_map(indices: torch.Tensor) -> torch.Tensor:
    images = []
    for idx in indices.view(-1).tolist():
        img = Image.new("RGB", (28, 28), (250, 250, 250))
        draw = ImageDraw.Draw(img)
        draw.text((12, 9), str(idx), (0, 0, 0))
        images.append(to_torch(np.array(img).transpose([2, 0, 1])))
    return torch.stack(images).float()
