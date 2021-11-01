import torch

from torch import nn
from typing import Any
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.feature_extraction import create_feature_extractor

from .register import backbone_info_dict
from .....types import tensor_dict_type
from .....constants import LATENT_KEY
from .....misc.toolkit import set_requires_grad


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
        try:
            self.core = create_feature_extractor(core, return_nodes=self.return_nodes)
        except:
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
