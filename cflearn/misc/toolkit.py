import io
import os
import sys
import copy
import json
import math
import torch
import random
import hashlib
import argparse
import urllib.request

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from PIL import ImageDraw
from torch import Tensor
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from typing import ContextManager
from zipfile import ZipFile
from collections import defaultdict
from collections import OrderedDict
from torch.optim import Optimizer
from cftool.misc import prod
from cftool.misc import print_info
from cftool.misc import print_warning
from cftool.misc import check_requires
from cftool.misc import shallow_copy_dict
from cftool.misc import truncate_string_to_length
from cftool.misc import DownloadProgressBar
from cftool.array import to_torch
from cftool.array import is_string
from cftool.array import to_standard
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type
from safetensors.torch import load_file

from ..types import data_type
from ..types import param_type
from ..types import sample_weights_type
from ..constants import INPUT_KEY
from ..constants import WORKSPACE_ENVIRON_KEY
from ..parameters import OPT

try:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure as Figure
except:
    plt = Figure = None
try:
    from onnxruntime import InferenceSession
except:
    InferenceSession = None
try:
    import cv2
except:
    cv2 = None


# general


min_seed_value = np.iinfo(np.uint32).min
max_seed_value = np.iinfo(np.uint32).max


def new_seed() -> int:
    return random.randint(min_seed_value, max_seed_value)


def seed_everything(seed: int) -> int:
    if not min_seed_value <= seed <= max_seed_value:
        msg = f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}"
        print_warning(msg)
        seed = new_seed()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def _get_environ_workspace() -> Optional[str]:
    return os.environ.get(WORKSPACE_ENVIRON_KEY)


def _set_environ_workspace(workspace: str) -> None:
    os.environ[WORKSPACE_ENVIRON_KEY] = workspace


def check_is_ci() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", type=int, default=0)
    args = parser.parse_args()
    return bool(args.ci)


class FileInfo(NamedTuple):
    sha: str
    st_size: int
    download_url: Optional[str] = None


def check_available(tag: str, repo: str, name: str) -> Optional[FileInfo]:
    with open(os.path.join(os.path.dirname(__file__), "available.json"), "r") as f:
        available = json.load(f)
    if repo != "pretrained-models":
        return FileInfo("", 0)
    info = available[tag].get(name)
    return None if info is None else FileInfo(**info)


def _get_file_size(path: str) -> int:
    return os.stat(path).st_size


def _get_file_info(path: str) -> FileInfo:
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return FileInfo(sha, _get_file_size(path))


def _check_sha(path: str, tgt_sha: str) -> bool:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest() == tgt_sha


def download(
    tag: str,
    repo: str,
    name: str,
    root: str,
    extension: str,
    *,
    check_sha: bool = False,
    remove_zip: bool = True,
) -> str:
    info = check_available(tag, repo, name)
    if info is None:
        raise ValueError(f"'{name}' is currently not available at '{tag}'")
    os.makedirs(root, exist_ok=True)
    file = f"{name}.{extension}"
    path = os.path.join(root, file)
    is_zip = extension == "zip"
    zip_folder_path = os.path.join(root, name)
    if is_zip and os.path.isdir(zip_folder_path):
        return zip_folder_path
    fmt = "cache file is detected but {}, it will be re-downloaded"
    if not is_zip and os.path.isfile(path):
        if os.stat(path).st_size != info.st_size:
            print_warning(fmt.format("st_size is not correct"))
        else:
            if not check_sha or _check_sha(path, info.sha):
                return path
            print_warning(fmt.format("sha is not correct"))
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=name) as t:
        if info.download_url is not None:
            url = info.download_url
        else:
            prefix = f"https://github.com/carefree0910/{repo}/releases/download/{tag}/"
            url = f"{prefix}{file}"
        urllib.request.urlretrieve(
            url,
            filename=path,
            reporthook=t.update_to,
        )
    if not is_zip:
        return path
    with ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(zip_folder_path)
    if remove_zip:
        os.remove(path)
    return zip_folder_path


def download_tokenizer(
    name: str,
    *,
    root: Optional[str] = None,
) -> str:
    if root is None:
        root = os.path.join(OPT.cache_dir, "tokenizers")
    return download("tokenizers", "pretrained-models", name, root, "pkl")


def download_model(
    name: str,
    *,
    root: Optional[str] = None,
) -> str:
    if root is None:
        root = os.path.join(OPT.cache_dir, "models")
    return download("checkpoints", "pretrained-models", name, root, "pt")


def download_reference(
    name: str,
    *,
    root: Optional[str] = None,
) -> str:
    if root is None:
        root = os.path.join(OPT.cache_dir, "reference")
    return download("reference", "pretrained-models", name, root, "pt")


def download_static(
    name: str,
    *,
    extension: str,
    root: Optional[str] = None,
) -> str:
    if root is None:
        root = os.path.join(OPT.cache_dir, "static")
    return download("static", "pretrained-models", name, root, extension)


def download_dataset(
    name: str,
    *,
    root: str = os.getcwd(),
    remove_zip: bool = True,
) -> str:
    return download("latest", "datasets", name, root, "zip", remove_zip=remove_zip)


def get_compatible_name(
    tag: str,
    repo: str,
    name: str,
    versions: List[Tuple[int, int]],
    *,
    bc: bool = False,
) -> str:
    version_info = sys.version_info
    version = None
    if bc:
        tgt_versions = list(
            filter(
                lambda ver: version_info.major < ver[0] or version_info.minor < ver[1],
                versions,
            )
        )
        if tgt_versions is not None:
            version = max(tgt_versions)
    if not bc:
        tgt_versions = list(
            filter(
                lambda ver: version_info.major > ver[0] or version_info.minor >= ver[1],
                versions,
            )
        )
        if tgt_versions is not None:
            version = max(tgt_versions)
    if version is not None:
        compatible_name = f"{name}_{version[0]}.{version[1]}"
        if check_available(tag, repo, compatible_name):
            name = compatible_name
        else:
            print_warning(
                f"compatible name '{compatible_name}' is not available "
                f"on the server, will use the original name ({name}) instead"
            )
    return name


def show_or_save(
    export_path: Optional[str],
    fig: Optional[Figure] = None,
    **kwargs: Any,
) -> None:
    """
    Utility function to deal with figure.

    Parameters
    ----------
    export_path : {None, str}
    * If None, the figure will be shown.
    * If str, it represents the path where the figure should be saved to.
    fig : {None, plt.Figure}
    * If None, default figure contained in plt will be executed.
    * If plt.figure, it will be executed

    """

    if plt is None:
        raise ValueError("`matplotlib` is needed for `show_or_save`")
    if export_path is None:
        fig.show(**kwargs) if fig is not None else plt.show(**kwargs)
    else:
        if fig is not None:
            fig.savefig(export_path)
        else:
            plt.savefig(export_path, **kwargs)
    plt.close()


def show_or_return(return_canvas: bool) -> Union[None, np.ndarray]:
    """
    Utility function to deal with current plt.

    Parameters
    ----------
    return_canvas : bool, whether return canvas or not.

    """

    if plt is None:
        raise ValueError("`matplotlib` is needed for `show_or_return`")
    if not return_canvas:
        plt.show()
        return None

    buffer_ = io.BytesIO()
    plt.savefig(buffer_, format="png")
    plt.close()
    buffer_.seek(0)
    image = Image.open(buffer_)
    canvas = np.asarray(image)[..., :3]
    buffer_.close()
    return canvas


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
        if plt is None:
            raise ValueError("`matplotlib` is needed for `visualize`")
        if show_or_save is None:
            raise ValueError("`carefree-ml` is needed for `visualize`")
        n = 1000
        x = np.linspace(0, 1, n)
        y = self(n, 0)
        if isinstance(y, tuple):
            y = y[0]
        plt.figure()
        plt.plot(x, y)
        show_or_save(export_path)


# dl


pt2_sdp_attn = getattr(F, "scaled_dot_product_attention", None)


def sdp_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    training: bool,
    mask: Optional[Tensor] = None,
    dropout: Optional[float] = None,
) -> Tensor:
    if pt2_sdp_attn is not None:
        dropout = dropout if training else None
        dropout = 0.0 if dropout is None else dropout
        return pt2_sdp_attn(q, k, v, mask, dropout)
    raw_weights = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
    if mask is not None:
        raw_weights.masked_fill_(~mask, float("-inf"))
    weights = F.softmax(raw_weights, dim=-1)
    if training and dropout is not None and 0.0 < dropout < 1.0:
        weights = F.dropout(weights, dropout)
    return weights @ v


def get_tensors(inp: Union[str, tensor_dict_type]) -> tensor_dict_type:
    if isinstance(inp, str):
        if inp.endswith(".safetensors"):
            inp = load_file(inp)
        else:
            inp = torch.load(inp, map_location="cpu")
    if "state_dict" in inp:
        inp = inp["state_dict"]
    return shallow_copy_dict(inp)


def get_dtype(m: nn.Module) -> torch.dtype:
    params = list(m.parameters())
    return torch.float32 if not params else params[0].dtype


def get_device(m: nn.Module) -> torch.device:
    params = list(m.parameters())
    return torch.device("cpu") if not params else params[0].device


def get_clones(
    module: nn.Module,
    n: int,
    *,
    return_list: bool = False,
) -> Union[nn.ModuleList, List[nn.Module]]:
    module_list = [module]
    for _ in range(n - 1):
        module_list.append(copy.deepcopy(module))
    if return_list:
        return module_list
    return nn.ModuleList(module_list)


def get_torch_device(device: Union[int, str, torch.device]) -> torch.device:
    if isinstance(device, (int, str)):
        device = torch.device(device)
    return device


def empty_cuda_cache(device: Union[int, str, torch.device]) -> None:
    device = get_torch_device(device)
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


def is_cpu(device: Union[int, str, torch.device]) -> bool:
    return get_torch_device(device).type == "cpu"


def np_batch_to_tensor(np_batch: np_dict_type) -> tensor_dict_type:
    return {
        k: v if not isinstance(v, np.ndarray) or is_string(v) else to_torch(v)
        for k, v in np_batch.items()
    }


def tensor_batch_to_np(tensor_batch: np_dict_type) -> np_dict_type:
    return {
        k: v if not isinstance(v, Tensor) else v.cpu().numpy()
        for k, v in tensor_batch.items()
    }


def safe_clip_(net: Tensor) -> None:
    finfo = torch.finfo(net.dtype)
    net.clamp_(finfo.min, finfo.max)


def fix_denormal_states(
    states: tensor_dict_type,
    *,
    eps: float = 1.0e-32,
    verbose: bool = False,
) -> tensor_dict_type:
    new_states = shallow_copy_dict(states)
    num_total = num_denormal_total = 0
    for k, v in states.items():
        if not v.is_floating_point():
            continue
        num_total += v.numel()
        denormal = (v != 0) & (v.abs() < eps)
        num_denormal = denormal.sum().item()
        num_denormal_total += num_denormal
        if num_denormal > 0:
            new_states[k][denormal] = v.new_zeros(num_denormal)
    if verbose:
        print_info(f"denormal ratio : {num_denormal_total / num_total:8.6f}")
    return new_states


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


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    # make sure that the module will never go back to `train` mode
    module.train = lambda m, mode=True: m  # type: ignore
    set_requires_grad(module, False)
    return module


def scheduler_requires_metric(scheduler: Any) -> bool:
    return check_requires(scheduler.step, "metrics")


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
        for p in module_.parameters():
            local_num_params = int(round(prod(p.data.shape)))
            num_params += local_num_params
            if p.requires_grad:
                num_trainable_params += local_num_params
        return num_params, num_trainable_params

    def register_hook(module: nn.Module) -> None:
        def inject_output_shape(output: Any, res: Dict[int, Any]) -> None:
            idx = 0 if not res else max(res)
            if isinstance(output, Tensor):
                o_shape = list(output.shape)
                if o_shape:
                    o_shape[0] = -1
                res[idx + 1] = o_shape
                return
            if isinstance(output, (list, tuple)):
                o_res = res[idx + 1] = {}
                for o in output:
                    inject_output_shape(o, o_res)

        def hook(module_: nn.Module, inp: Any, output: Any) -> None:
            m_name = module_names.get(module_)
            if m_name is None:
                return

            if not inp:
                return
            inp = inp[0]
            if not isinstance(inp, Tensor):
                return

            m_dict: OrderedDict[str, Any] = OrderedDict()
            m_dict["input_shape"] = list(inp.shape)
            if len(m_dict["input_shape"]) > 0:
                m_dict["input_shape"][0] = -1
            output_shape_res = m_dict["output_shape"] = {}
            inject_output_shape(output, output_shape_res)

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
            model.run(0, sample_batch)
        else:
            model.summary_forward(sample_batch)  # type: ignore
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
    messages.append(line_format.format(*headers))
    messages.append("-" * line_length)
    total_output = 0
    for layer, layer_summary in summary_dict.items():
        layer_name = "-".join(layer.split("-")[:-1])
        valid_layer_name = layer_name.strip()
        num_spaces = len(layer_name) - len(valid_layer_name)
        valid_layer_name = truncate_string_to_length(valid_layer_name, 30 - num_spaces)
        layer_name = " " * num_spaces + valid_layer_name
        if layer_summary is None:
            messages.append(line_format.format(layer_name, "", "", ""))
        else:
            is_title = True
            all_output_shapes: List[List[int]] = []

            def _inject(output_shape_item: Dict[int, Any], prefix: str) -> None:
                only_one = len(output_shape_item) == 1
                for i, idx in enumerate(sorted(output_shape_item)):
                    if not prefix and only_one:
                        idx_prefix = ""
                    else:
                        idx_prefix = f"{prefix}{idx}."
                    value = output_shape_item[idx]
                    if isinstance(value, dict):
                        _inject(value, idx_prefix)
                        continue
                    output_shape_str = f"{idx_prefix} {str(value):>16s}"
                    ntp_str = "{0:,}".format(layer_summary["num_trainable_params"])
                    nonlocal is_title
                    messages.append(
                        line_format.format(
                            layer_name if is_title else "",
                            str(layer_summary["input_shape"]) if is_title else "",
                            output_shape_str,
                            ntp_str if is_title else "",
                        )
                    )
                    is_title = False
                    all_output_shapes.append(value)

            _inject(layer_summary["output_shape"], "")
            for shape in all_output_shapes:
                total_output += prod(shape)

    total_params, trainable_params = _get_param_counts(model)
    # assume 4 bytes/number (float on cuda).
    x_batch = sample_batch[INPUT_KEY]
    get_size = lambda t: abs(prod(t.shape[1:]) * 4.0 / (1024**2.0))
    if not isinstance(x_batch, list):
        x_batch = [x_batch]
    total_input_size = sum(map(get_size, x_batch))
    # x2 for gradients
    total_output_size = abs(2.0 * total_output * 4.0 / (1024**2.0))
    total_params_size = abs(total_params * 4.0 / (1024**2.0))
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


def is_local_rank_0() -> bool:
    ddp_info = get_ddp_info()
    return ddp_info is None or ddp_info.local_rank == 0


def get_world_size() -> int:
    ddp_info = get_ddp_info()
    return 1 if ddp_info is None else ddp_info.world_size


class toggle_optimizer:
    """
    Help focusing gradients on specific optimizer and recovering previous states

    This is a context controller for requiring and only requiring grads for parameters
    of the given optimizer at the beginning, and back to previous grads requiring states
    at the end.

    Examples
    --------
    >>> module = nn.Module()
    >>> optimizer = torch.optim.Adam()
    >>> with toggle_optimizer(module, optimizer):
    >>>     pass  # do something

    """

    def __init__(self, m: nn.Module, optimizer: Optimizer, *, enabled: bool = True):
        self.m = m
        self.optimizer = optimizer
        self.enabled = enabled
        self.requires_grad: Dict[str, bool] = {}

    def __enter__(self) -> None:
        if not self.enabled:
            return
        self.requires_grad = {k: p.requires_grad for k, p in self.m.named_parameters()}
        for p in self.m.parameters():
            p.requires_grad = False
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                p.requires_grad = True

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.enabled:
            return
        for k, p in self.m.named_parameters():
            requires_grad = self.requires_grad.get(k)
            if requires_grad is not None:
                p.requires_grad = requires_grad


class mode_context:
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


class no_grad_context(torch.no_grad):
    def __init__(self, *, enabled: bool):
        super().__init__()
        self.enabled = enabled

    def __enter__(self) -> None:
        if not self.enabled:
            return
        super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.enabled:
            return
        super().__exit__(exc_type, exc_val, exc_tb)


class Initializer:
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
            print_warning(f"'{name}' initializer is already defined")
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
            print_warning(
                "invalid ratio for truncated normal : "
                f"{invalid.to(torch.float32).mean():8.6f}, it might cause by "
                f"too little epoch ({epoch}) or too small tolerance ({tol})",
            )
        with torch.no_grad():
            param.data.copy_(weight_base.reshape(param.shape))
            param.data.mul_(std).add_(mean)

    def orthogonal(self, param: param_type) -> None:
        gain = self.config.setdefault("gain", 1.0)
        nn.init.orthogonal_(param.data, gain)


class ONNX:
    def __init__(self, onnx_path: str):
        if InferenceSession is None:
            msg = "`ONNX` is not available when `onnxruntime` is not installed"
            raise ValueError(msg)
        self.ort_session = InferenceSession(onnx_path)
        self.output_names = [node.name for node in self.ort_session.get_outputs()]

    def predict(self, new_inputs: np_dict_type) -> np_dict_type:
        if self.ort_session is None:
            raise ValueError("`onnx_path` is not provided")
        ort_inputs = {
            node.name: to_standard(new_inputs[node.name])
            for node in self.ort_session.get_inputs()
        }
        return dict(zip(self.output_names, self.ort_session.run(None, ort_inputs)))


def gradient_checkpoint(func: Callable, inputs: Any, params: Any, enabled: bool) -> Any:
    if not enabled:
        return func(*inputs)
    args = tuple(inputs) + tuple(params)
    return GradientCheckpointFunction.apply(func, len(inputs), *args)


class GradientCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        run_function, length, *args = args  # type: ignore
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            grad_outputs,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


# ml


def to_2d(arr: data_type) -> data_type:
    if arr is None or isinstance(arr, str):
        return None
    if isinstance(arr, np.ndarray):
        return arr.reshape([len(arr), -1])
    if isinstance(arr[0], list):
        return arr
    return [[elem] for elem in arr]  # type: ignore


# cv


def auto_num_layers(
    img_size: int,
    min_size: int = 4,
    target_layers: Optional[int] = 4,
    *,
    use_stride: bool = False,
) -> int:
    fn = math.ceil if use_stride else math.floor
    max_layers = fn(math.log2(img_size / min_size))
    if target_layers is None:
        return max_layers
    return max(2, min(target_layers, max_layers))


def slerp(
    x1: torch.Tensor,
    x2: torch.Tensor,
    r1: Union[float, torch.Tensor],
    r2: Optional[Union[float, torch.Tensor]] = None,
    *,
    dot_threshold: float = 0.9995,
) -> torch.Tensor:
    if r2 is None:
        r2 = 1.0 - r1
    b, *shape = x1.shape
    x1 = x1.view(b, -1)
    x2 = x2.view(b, -1)
    low_norm = x1 / torch.norm(x1, dim=1, keepdim=True)
    high_norm = x2 / torch.norm(x2, dim=1, keepdim=True)
    dot = (low_norm * high_norm).sum(1)
    overflow_mask = dot > dot_threshold
    out = torch.zeros_like(x1)
    out[overflow_mask] = r1 * x1 + r2 * x2
    normal_mask = ~overflow_mask
    omega = torch.acos(dot[normal_mask])
    so = torch.sin(omega)
    x1_part = (torch.sin(r1 * omega) / so).unsqueeze(1) * x1
    x2_part = (torch.sin(r2 * omega) / so).unsqueeze(1) * x2
    out[normal_mask] = x1_part + x2_part
    return out.view(b, *shape)


def interpolate(
    src: Tensor,
    *,
    mode: str = "nearest",
    factor: Optional[Union[float, Tuple[float, float]]] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    anchor: Optional[Tensor] = None,
    determinate: bool = False,
    **kwargs: Any,
) -> Tensor:
    if "linear" in mode or mode == "bicubic":
        kwargs.setdefault("align_corners", False)
    c, h, w = src.shape[1:]
    if determinate:
        c, h, w = map(int, [c, h, w])
    if factor is not None:
        template = "`{}` will take no affect because `factor` is provided"
        if size is not None:
            print_warning(template.format("size"))
        if anchor is not None:
            print_warning(template.format("anchor"))
        if factor == 1.0 or factor == (1.0, 1.0):
            return src
        if not determinate:
            return F.interpolate(
                src,
                mode=mode,
                scale_factor=factor,
                recompute_scale_factor=True,
                **kwargs,
            )
        if not isinstance(factor, tuple):
            factor = factor, factor
        size = tuple(map(int, map(round, [h * factor[0], w * factor[1]])))  # type: ignore
    if size is None:
        if anchor is None:
            raise ValueError("either `size` or `anchor` should be provided")
        size = anchor.shape[2:]
        if determinate:
            size = tuple(map(int, size))  # type: ignore
    if not isinstance(size, tuple):
        size = size, size
    if h == size[0] and w == size[1]:
        return src
    net = F.interpolate(src, size=size, mode=mode, **kwargs)
    if not determinate:
        return net
    return net.view(-1, c, *size)


def mean_std(
    latent_map: Tensor,
    eps: float = 1.0e-5,
    *,
    determinate: bool = False,
) -> Tuple[Tensor, Tensor]:
    c, h, w = latent_map.shape[1:]
    if determinate:
        c, h, w = map(int, [c, h, w])
    spatial_dim = h * w
    latent_var = latent_map.view(-1, c, spatial_dim).var(dim=2) + eps
    latent_std = latent_var.sqrt().view(-1, c, 1, 1)
    latent_mean = latent_map.view(-1, c, spatial_dim).mean(dim=2).view(-1, c, 1, 1)
    return latent_mean, latent_std


def adain_with_params(
    src: Tensor,
    mean: Tensor,
    std: Tensor,
    *,
    determinate: bool = False,
) -> Tensor:
    src_mean, src_std = mean_std(src, determinate=determinate)
    src_normalized = (src - src_mean) / src_std
    return src_normalized * std + mean


def adain_with_tensor(src: Tensor, tgt: Tensor, *, determinate: bool = False) -> Tensor:
    tgt_mean, tgt_std = mean_std(tgt, determinate=determinate)
    return adain_with_params(src, tgt_mean, tgt_std, determinate=determinate)


def make_indices_visualization_map(indices: Tensor) -> Tensor:
    images = []
    for idx in indices.view(-1).tolist():
        img = Image.new("RGB", (28, 28), (250, 250, 250))
        draw = ImageDraw.Draw(img)
        draw.text((12, 9), str(idx), (0, 0, 0))
        images.append(to_torch(np.array(img).transpose([2, 0, 1])))
    return torch.stack(images).float()
