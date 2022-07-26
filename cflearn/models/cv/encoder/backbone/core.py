import torch

from torch import nn
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from collections import OrderedDict
from cftool.types import tensor_dict_type

from .register import backbone_info_dict
from .....constants import LATENT_KEY
from .....misc.toolkit import set_requires_grad


def inject_all_named_children(m: nn.Module, rs: List, prefix: Optional[str]) -> None:
    for name, mm in m.named_children():
        if prefix is not None:
            name = f"{prefix}.{name}"
        if not list(mm.named_children()):
            rs.append((name, mm))
        else:
            inject_all_named_children(mm, rs, name)


def check_named_children(
    named_children: List[Tuple[str, nn.Module]],
    return_layers: Dict[str, str],
) -> List[str]:
    all_names = set(name for name, _ in named_children)
    invalid_names = []
    for layer in return_layers:
        if layer not in all_names:
            invalid_names.append(layer)
    return invalid_names


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        all_named_children = list(model.named_children())
        invalid_names = check_named_children(all_named_children, return_layers)
        if invalid_names:
            all_named_children = []
            inject_all_named_children(model, all_named_children, None)
            all_names = set(name for name, _ in all_named_children)
            invalid_names = []
            for layer in return_layers:
                if layer not in all_names:
                    invalid_names.append(layer)
            if invalid_names:
                invalid_msg = ", ".join(invalid_names)
                raise ValueError(f"following layers: {invalid_msg} are not presented")
        name_mapping = {k: str(i) for i, (k, _) in enumerate(all_named_children)}
        all_named_children = [(name_mapping[k], m) for k, m in all_named_children]
        return_layers = {name_mapping[k]: v for k, v in return_layers.items()}
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in all_named_children:
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        out = OrderedDict()
        for name, module in self.items():
            net = module(net)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = net
        return out


class Backbone(nn.Module):
    def __init__(
        self,
        name: str = "resnet101",
        *,
        pretrained: bool = True,
        requires_grad: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        backbone_info = backbone_info_dict.get(name)
        if backbone_info is None:
            raise ValueError(f"backbone '{name}' is not recognized")
        self.out_channels = backbone_info.out_channels
        self.latent_channels = self.out_channels[-1]
        core = backbone_info.fn(pretrained, **kwargs)
        self._original = [core]
        self.return_nodes = backbone_info.return_nodes
        self.core = IntermediateLayerGetter(core, self.return_nodes)
        set_requires_grad(self.core, requires_grad)
        stage_idx = set()
        for layer in self.return_nodes.values():
            if layer.startswith("stage"):
                stage_idx.add(int(layer.split("_")[0][-1]))
        self.num_downsample = len(stage_idx) - int(bool(0 in stage_idx))

    @property
    def original(self) -> nn.Module:
        return self._original[0]

    def forward(self, net: torch.Tensor) -> tensor_dict_type:
        rs = self.core(net)
        rs[LATENT_KEY] = list(rs.values())[-1]
        return rs


__all__ = [
    "Backbone",
]
