import os
import json
import math
import time
import torch
import shutil
import inspect
import logging
import torchvision
import urllib.request

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm
from torch import Tensor
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
from typing import NamedTuple
from typing import ContextManager
from zipfile import ZipFile
from argparse import Namespace
from datetime import datetime
from datetime import timedelta
from collections import defaultdict
from collections import OrderedDict
from torch.optim import Optimizer
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
from ..constants import CACHE_DIR
from ..constants import INPUT_KEY
from ..constants import TIME_FORMAT
from ..constants import INFO_PREFIX
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


def download(
    name: str,
    root: str,
    prefix: str,
    extension: str,
    force_download: bool,
    remove_zip: Optional[bool],
    extract_zip: bool,
) -> str:
    os.makedirs(root, exist_ok=True)
    file = f"{name}.{extension}"
    path = os.path.join(root, file)
    is_zip = extension == "zip"
    zip_folder_path = os.path.join(root, name)
    exists = os.path.isdir(zip_folder_path) if is_zip else os.path.isfile(path)
    if exists and not force_download:
        return zip_folder_path if is_zip else path
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=name) as t:
        urllib.request.urlretrieve(
            f"{prefix}{file}",
            filename=path,
            reporthook=t.update_to,
        )
    if not is_zip:
        return path
    if extract_zip:
        with ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(zip_folder_path)
    if remove_zip is None:
        remove_zip = extract_zip
    if remove_zip:
        if extract_zip:
            os.remove(path)
        else:
            print(
                f"{WARNING_PREFIX}zip file is not extracted, "
                f"so we will not remove it!"
            )
    return zip_folder_path


def download_data_info(
    name: str,
    *,
    root: str = os.path.join(CACHE_DIR, "data_info"),
    prefix: str = "https://github.com/carefree0910/pretrained-models/releases/download/data_info/",
    force_download: bool = False,
) -> str:
    return download(name, root, prefix, "json", force_download, None, False)


def download_model(
    name: str,
    *,
    root: str = os.path.join(CACHE_DIR, "models"),
    prefix: str = "https://github.com/carefree0910/pretrained-models/releases/download/checkpoints/",
    force_download: bool = False,
) -> str:
    return download(name, root, prefix, "pt", force_download, None, False)


def download_dataset(
    name: str,
    *,
    root: str = os.getcwd(),
    prefix: str = "https://github.com/carefree0910/datasets/releases/download/latest/",
    force_download: bool = False,
    remove_zip: Optional[bool] = None,
    extract_zip: bool = True,
) -> str:
    return download(name, root, prefix, "zip", force_download, remove_zip, extract_zip)


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

    @classmethod
    def check_subclass(cls, name: str) -> bool:
        return issubclass(cls.d[name], cls)


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


class DownloadProgressBar(tqdm):
    def update_to(
        self,
        b: int = 1,
        bsize: int = 1,
        tsize: Optional[int] = None,
    ) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


# dl


def fix_denormal_states(
    states: tensor_dict_type,
    *,
    verbose: bool = False,
) -> tensor_dict_type:
    new_states = shallow_copy_dict(states)
    num_total = num_denormal_total = 0
    for k, v in states.items():
        num_total += v.numel()
        denormal = (v != 0) & (v.abs() < 1.0e-32)
        num_denormal = denormal.sum().item()
        num_denormal_total += num_denormal
        if num_denormal > 0:
            new_states[k][denormal] = v.new_zeros(num_denormal)
    if verbose:
        print(f"{INFO_PREFIX}denormal ratio : {num_denormal_total / num_total:8.6f}")
    return new_states


def toggle_optimizer(m: nn.Module, optimizer: Optimizer) -> None:
    for param in m.parameters():
        param.requires_grad = False
    for group in optimizer.param_groups:
        for param in group["params"]:
            param.requires_grad = True


def to_torch(arr: np.ndarray) -> Tensor:
    return torch.from_numpy(to_standard(arr))


def to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def to_device(batch: tensor_dict_type, device: torch.device) -> tensor_dict_type:
    return {
        k: v.to(device)
        if isinstance(v, Tensor)
        else [vv.to(device) if isinstance(vv, Tensor) else vv for vv in v]
        if isinstance(v, list)
        else v
        for k, v in batch.items()
    }


def squeeze(arr: arr_type) -> arr_type:
    n = arr.shape[0]
    arr = arr.squeeze()
    if n == 1:
        arr = arr[None, ...]
    return arr


def softmax(arr: arr_type) -> arr_type:
    if isinstance(arr, Tensor):
        return F.softmax(arr, dim=1)
    logits = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(1, keepdims=True)


def has_batch_norms(m: nn.Module) -> bool:
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in m.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def inject_parameters(
    src: nn.Module,
    tgt: nn.Module,
    *,
    strict: Optional[bool] = None,
    src_filter_fn: Optional[Callable[[str], bool]] = None,
    tgt_filter_fn: Optional[Callable[[str], bool]] = None,
    custom_mappings: Optional[Dict[str, str]] = None,
    states_callback: Optional[Callable[[tensor_dict_type], tensor_dict_type]] = None,
) -> None:
    if strict is None:
        strict = tgt_filter_fn is None
    src_states = src.state_dict()
    tgt_states = tgt.state_dict()
    if src_filter_fn is not None:
        pop_keys = [key for key in src_states if not src_filter_fn(key)]
        for key in pop_keys:
            src_states.pop(key)
    if tgt_filter_fn is not None:
        pop_keys = [key for key in tgt_states if not tgt_filter_fn(key)]
        for key in pop_keys:
            tgt_states.pop(key)
    if states_callback is not None:
        src_states = states_callback(shallow_copy_dict(src_states))
    if len(src_states) != len(tgt_states):
        raise ValueError(f"lengths of states are not identical between {src} and {tgt}")
    new_states = OrderedDict()
    if custom_mappings is not None:
        for src_k, tgt_k in custom_mappings.items():
            new_states[tgt_k] = src_states.pop(src_k)
            tgt_states.pop(tgt_k)
    for (src_k, src_v), (tgt_k, tgt_v) in zip(src_states.items(), tgt_states.items()):
        if src_v.shape != tgt_v.shape:
            raise ValueError(
                f"shape of {src_k} ({list(src_v.shape)}) is not identical with "
                f"shape of {tgt_k} ({list(tgt_v.shape)})"
            )
        new_states[tgt_k] = src_v
    tgt.load_state_dict(new_states, strict=strict)


class Diffs(NamedTuple):
    names1: List[str]
    names2: List[str]
    diffs: List[Tensor]


def sorted_param_diffs(m1: nn.Module, m2: nn.Module) -> Diffs:
    names1, params1 = list(zip(*m1.named_parameters()))
    names2, params2 = list(zip(*m2.named_parameters()))
    if len(params1) != len(params2):
        raise ValueError(f"lengths of params are not identical between {m1} and {m2}")
    diffs = []
    for p1, p2 in zip(params1, params2):
        (p1, _), (p2, _) = map(torch.sort, [p1.view(-1), p2.view(-1)])
        diffs.append(torch.abs(p1.data - p2.data))
    return Diffs(names1, names2, diffs)


def get_gradient(
    y: Tensor,
    x: Tensor,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> Union[Tensor, Tuple[Tensor, ...]]:
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
            if not isinstance(inp, Tensor):
                return
            if isinstance(output, (list, tuple)):
                for element in output:
                    if not isinstance(element, Tensor):
                        return
            elif not isinstance(output, Tensor):
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
    with eval_context(model, use_grad=None):
        if not hasattr(model, "summary_forward"):
            model(0, sample_batch)
        else:
            model.summary_forward(0, sample_batch)  # type: ignore
        for param in model.parameters():
            param.grad = None

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
    get_size = lambda t: abs(prod(t.shape[1:]) * 4.0 / (1024 ** 2.0))
    if not isinstance(x_batch, list):
        x_batch = [x_batch]
    total_input_size = sum(map(get_size, x_batch))
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


class DDPInfo(NamedTuple):
    rank: int
    world_size: int
    local_rank: int


def get_ddp_info() -> Optional[DDPInfo]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return DDPInfo(rank, world_size, local_rank)
    return None


def get_world_size() -> int:
    ddp_info = get_ddp_info()
    return 1 if ddp_info is None else ddp_info.world_size


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
        use_inference: Optional[bool] = None,
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
        if use_inference is None:
            self._inference_context: Optional[ContextManager] = None
        else:
            self._inference_context = torch.inference_mode(use_inference)

    def __enter__(self) -> None:
        if self._to_train is not None:
            self._module.train(mode=self._to_train)
        if self._grad_context is not None:
            self._grad_context.__enter__()
        if self._inference_context is not None:
            self._inference_context.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._to_train is not None:
            self._module.train(mode=self._training)
        if self._inference_context is not None:
            self._inference_context.__exit__(exc_type, exc_val, exc_tb)
        if self._grad_context is not None:
            self._grad_context.__exit__(exc_type, exc_val, exc_tb)
        for p, v in self._cache.items():
            p.requires_grad_(v)


class train_context(mode_context):
    """
    Useful when we need to get gradients with our PyTorch model during evaluating.
    """

    def __init__(self, module: nn.Module, *, use_grad: bool = True):
        super().__init__(module, to_train=True, use_grad=use_grad, use_inference=False)


class eval_context(mode_context):
    """
    Useful when we need to predict something with our PyTorch model during training.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        use_grad: Optional[bool] = False,
        use_inference: Optional[bool] = None,
    ):
        if use_inference is None and use_grad is not None:
            use_inference = not use_grad
        super().__init__(
            module,
            to_train=False,
            use_grad=use_grad,
            use_inference=use_inference,
        )


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
    predictions: Tensor,
    target: Tensor,
    weights: Optional[Tensor] = None,
    *,
    get_diagonal: bool = False,
) -> Tensor:
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


def auto_num_layers(
    img_size: int,
    min_size: int = 4,
    target_layers: Optional[int] = 4,
) -> int:
    max_layers = math.floor(math.log2(img_size / min_size))
    if target_layers is None:
        return max_layers
    return max(2, min(target_layers, max_layers))


def slerp(
    x1: torch.Tensor,
    x2: torch.Tensor,
    r1: Union[float, torch.Tensor],
    r2: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
    low_norm = x1 / torch.norm(x1, dim=1, keepdim=True)
    high_norm = x2 / torch.norm(x2, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    if r2 is None:
        r2 = 1.0 - r1
    x1_part = (torch.sin(r1 * omega) / so).unsqueeze(1) * x1
    x2_part = (torch.sin(r2 * omega) / so).unsqueeze(1) * x2
    return x1_part + x2_part


def interpolate(
    src: Tensor,
    *,
    mode: str = "nearest",
    factor: Optional[Union[float, Tuple[float, float]]] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    anchor: Optional[Tensor] = None,
    **kwargs: Any,
) -> Tensor:
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
    if factor == 1.0 or factor == (1.0, 1.0):
        return src
    return F.interpolate(
        src,
        mode=mode,
        scale_factor=factor,
        recompute_scale_factor=True,
        **kwargs,
    )


def mean_std(latent_map: Tensor, eps: float = 1.0e-5) -> Tuple[Tensor, Tensor]:
    n, c = latent_map.shape[:2]
    latent_var = latent_map.view(n, c, -1).var(dim=2) + eps
    latent_std = latent_var.sqrt().view(n, c, 1, 1)
    latent_mean = latent_map.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return latent_mean, latent_std


def adain_with_params(src: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    src_mean, src_std = mean_std(src)
    src_normalized = (src - src_mean) / src_std
    return src_normalized * std + mean


def adain_with_tensor(src: Tensor, tgt: Tensor) -> Tensor:
    tgt_mean, tgt_std = mean_std(tgt)
    return adain_with_params(src, tgt_mean, tgt_std)


def make_grid(arr: arr_type, n_row: Optional[int] = None) -> Tensor:
    if isinstance(arr, np.ndarray):
        arr = to_torch(arr)
    if n_row is None:
        n_row = math.ceil(math.sqrt(len(arr)))
    return torchvision.utils.make_grid(arr, n_row)


def save_images(arr: arr_type, path: str, n_row: Optional[int] = None) -> None:
    if isinstance(arr, np.ndarray):
        arr = to_torch(arr)
    if n_row is None:
        n_row = math.ceil(math.sqrt(len(arr)))
    torchvision.utils.save_image(arr, path, normalize=True, nrow=n_row)


def iou(logits: arr_type, labels: arr_type) -> arr_type:
    is_torch = isinstance(logits, Tensor)
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


def clip_normalize(arr: arr_type) -> arr_type:
    fn = np if isinstance(arr, np.ndarray) else torch
    if arr.dtype == fn.uint8:
        return arr
    return fn.clip(arr, 0.0, 1.0)


def min_max_normalize(arr: arr_type, *, global_norm: bool = True) -> arr_type:
    eps = 1.0e-8
    if global_norm:
        arr_min, arr_max = arr.min(), arr.max()
        return (arr - arr_min) / max(eps, arr_max - arr_min)
    if isinstance(arr, np.ndarray):
        arr_min, arr_max = arr.min(axis=0), arr.max(axis=0)
        diff = np.maximum(eps, arr_max - arr_min)
    else:
        arr_min, arr_max = arr.min(dim=0).values, arr.max(dim=0).values
        diff = torch.clip(arr_max - arr_min, max=eps)
    return (arr - arr_min) / diff


def quantile_normalize(
    arr: arr_type,
    *,
    q: float = 0.01,
    global_norm: bool = True,
) -> arr_type:
    eps = 1.0e-8
    # quantiles
    if isinstance(arr, Tensor):
        kw = {"dim": 0}
        quantile_fn = torch.quantile
    else:
        kw = {"axis": 0}
        quantile_fn = np.quantile
    if global_norm:
        arr_min = quantile_fn(arr, q)
        arr_max = quantile_fn(arr, 1.0 - q)
    else:
        arr_min = quantile_fn(arr, q, **kw)
        arr_max = quantile_fn(arr, 1.0 - q, **kw)
    # diff
    if global_norm:
        diff = max(eps, arr_max - arr_min)
    else:
        if isinstance(arr, Tensor):
            diff = torch.clamp(arr_max - arr_min, min=eps)
        else:
            diff = np.maximum(eps, arr_max - arr_min)
    arr = arr.clip(arr_min, arr_max)
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


def make_indices_visualization_map(indices: Tensor) -> Tensor:
    images = []
    for idx in indices.view(-1).tolist():
        img = Image.new("RGB", (28, 28), (250, 250, 250))
        draw = ImageDraw.Draw(img)
        draw.text((12, 9), str(idx), (0, 0, 0))
        images.append(to_torch(np.array(img).transpose([2, 0, 1])))
    return torch.stack(images).float()
