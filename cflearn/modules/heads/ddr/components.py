import torch

from torch import Tensor
from typing import Any
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from torch.nn import Module
from torch.nn import ModuleList

from ...blocks import MLP
from ...blocks import CrossBase
from ...blocks import CrossBlock
from ...blocks import ConditionalBlocks
from ...blocks import MonotonousMapping
from ....types import tensor_tuple_type


def transition_builder(dim: int) -> Module:
    h_dim = int(dim // 2)
    return MonotonousMapping.make_couple(h_dim, h_dim, h_dim, "sigmoid", ascent=True)


def get_cond_mappings(
    condition_dim: int,
    cond_out_dim: Optional[int],
    num_units: List[int],
    to_latent: bool,
) -> ModuleList:
    cond_module = MLP.simple(
        condition_dim,
        cond_out_dim,
        num_units,
        bias=not to_latent,
        activation="glu",
    )
    return cond_module.mappings


responses_tuple_type = Tuple[List[Tensor], List[Tensor]]


class Pack(NamedTuple):
    net: Tensor
    cond: Tensor
    responses_tuple: responses_tuple_type


class MonoSplit(Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        ascent1: bool,
        ascent2: bool,
        to_latent: bool,
        builder: Callable[[int, int, bool], ConditionalBlocks],
    ) -> None:
        super().__init__()
        self.to_latent = to_latent
        self.m1 = builder(in_dim, out_dim, ascent1)
        self.m2 = builder(in_dim, out_dim, ascent2)

    def forward(
        self,
        net: Union[Tensor, tensor_tuple_type],
        cond: Union[Tensor, responses_tuple_type],
    ) -> Union[tensor_tuple_type, Pack]:
        if self.to_latent:
            return self.m1(net, cond), self.m2(net, cond)
        cond1: Union[Tensor, List[Tensor]]
        cond2: Union[Tensor, List[Tensor]]
        if isinstance(cond, tuple):
            cond1, cond2 = cond
        else:
            cond1 = cond2 = cond
        o1, o2 = self.m1(net[0], cond1), self.m2(net[1], cond2)
        merged_net = 0.5 * (o1.net + o2.net)
        merged_cond = 0.5 * (o1.cond + o2.cond)
        return Pack(merged_net, merged_cond, (o1.responses, o2.responses))


class MonoCross(CrossBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        centralize: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        kwargs["bias"] = not centralize
        self.mapping = MonotonousMapping(in_dim, out_dim, ascent=True, **kwargs)
        self.centralize = centralize

    def forward(self, net: Tensor, cond: Tensor) -> Tensor:
        crossed = self.mapping(net * cond.abs()) * torch.sigmoid(cond)
        if self.centralize:
            return crossed
        return crossed + cond

    def extra_repr(self) -> str:
        return f"(centralize): {self.centralize}"

    @classmethod
    def make(
        cls,
        dim: int,
        inner: bool,
        *,
        centralize: bool = True,
        **kwargs: Any,
    ) -> "MonoCross":
        out_dim = 1 if inner else dim
        return cls(dim, out_dim, centralize=centralize, **kwargs)


def get_q_cross_builder(to_latent: bool) -> Callable[[int], Module]:
    return lambda dim: MonoCross.make(dim, to_latent, centralize=to_latent)


def get_y_cross_builder(to_latent: bool) -> Callable[[int], Module]:
    return lambda dim: MonoCross.make(dim, to_latent, centralize=not to_latent)


def monotonous_builder(
    is_q: bool,
    ascent1: bool,
    ascent2: bool,
    to_latent: bool,
    num_layers: int,
    condition_dim: int,
    cond_mappings: Optional[ModuleList] = None,
) -> Callable[[int, int], Module]:
    def _core(
        in_dim: int,
        out_dim: int,
        ascent: bool,
    ) -> ConditionalBlocks:
        cond_out_dim: Optional[int]
        block_out_dim: Optional[int]
        if to_latent:
            num_units = [out_dim] * (num_layers + 1)
            cond_out_dim = block_out_dim = None
        else:
            num_units = [in_dim] * num_layers
            # only cdf will initialize cond mappings here
            # q will utilize median mappings defined outside
            cond_out_dim = 1
            # block : y_mul / cdf_mul
            block_out_dim = 1

        blocks = MonotonousMapping.stack(
            in_dim,
            block_out_dim,
            num_units,
            ascent=ascent,
            use_couple_bias=(is_q and not to_latent) or (not is_q and to_latent),
            activation="sigmoid" if to_latent else "tanh",
            return_blocks=True,
        )
        assert isinstance(blocks, list)

        if cond_mappings is not None:
            cond_mappings_ = cond_mappings
        else:
            cond_mappings_ = get_cond_mappings(
                condition_dim,
                cond_out_dim,
                num_units,
                to_latent,
            )

        if is_q:
            cross_builder = get_q_cross_builder(to_latent)
        else:
            cross_builder = get_y_cross_builder(to_latent)
        cond_mixtures = ModuleList(
            [
                CrossBlock(
                    unit,
                    False,
                    cross_builder=cross_builder,
                )
                for unit in num_units
            ]
        )
        return ConditionalBlocks(
            ModuleList(blocks),
            cond_mappings_,
            add_last=to_latent,
            detach_condition=not to_latent,
            cond_mixtures=cond_mixtures,
        )

    def _split_core(in_dim: int, out_dim: int) -> Module:
        if not to_latent:
            in_dim = int(in_dim // 2)
        else:
            out_dim = int(out_dim // 2)
        return MonoSplit(in_dim, out_dim, ascent1, ascent2, to_latent, _core)

    return _split_core


__all__ = [
    "get_cond_mappings",
    "monotonous_builder",
    "transition_builder",
    "responses_tuple_type",
    "Pack",
]
