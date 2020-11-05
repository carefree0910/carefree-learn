import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from cftool.misc import context_error_handler

from ...types import tensor_tuple_type
from ...misc.toolkit import switch_requires_grad
from ...modules.blocks import MLP
from ...modules.blocks import InvertibleBlock
from ...modules.blocks import MonotonousMapping
from ...modules.blocks import ConditionalBlocks
from ...modules.blocks import PseudoInvertibleBlock


def default_transition_builder(dim: int) -> nn.Module:
    h_dim = int(dim // 2)
    return MonotonousMapping.tanh_couple(h_dim, h_dim, h_dim, ascent=True)


def monotonous_builder(
    ascent1: bool,
    ascent2: bool,
    to_latent: bool,
    num_layers: int,
    condition_dim: int,
) -> Callable[[int, int], nn.Module]:
    def _core(in_dim: int, out_dim: int, ascent: bool) -> ConditionalBlocks:
        true_out_dim: Optional[int]
        if to_latent:
            num_units = [out_dim] * (num_layers + 1)
            true_out_dim = None
        else:
            num_units = [in_dim] * num_layers
            true_out_dim = out_dim

        blocks = MonotonousMapping.stack(
            in_dim, true_out_dim, num_units, ascent=ascent, return_blocks=True
        )
        assert isinstance(blocks, list)

        cond_module = MLP.simple(
            condition_dim,
            true_out_dim,
            num_units,
            activation="mish",
        )
        cond_mappings = cond_module.mappings

        return ConditionalBlocks(nn.ModuleList(blocks), cond_mappings)

    def _split_core(in_dim: int, out_dim: int) -> nn.Module:
        if not to_latent:
            in_dim = int(in_dim // 2)
        else:
            out_dim = int(out_dim // 2)

        class MonoSplit(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m1 = _core(in_dim, out_dim, ascent1)
                self.m2 = _core(in_dim, out_dim, ascent2)

            def forward(
                self,
                net: Union[Tensor, tensor_tuple_type],
                cond: Tensor,
            ) -> Union[Tensor, tensor_tuple_type]:
                if to_latent:
                    assert isinstance(net, Tensor)
                    return self.m1(net, cond), self.m2(net, cond)
                assert isinstance(net, tuple)
                return self.m1(net[0], cond) + self.m2(net[1], cond)

        return MonoSplit()

    return _split_core


class DDRCore(nn.Module):
    def __init__(
        self,
        in_dim: int,
        y_min: float,
        y_max: float,
        num_layers: Optional[int] = None,
        num_blocks: Optional[int] = None,
        latent_dim: Optional[int] = None,
        transition_builder: Optional[Callable[[int], nn.Module]] = None,
    ):
        super().__init__()
        # common
        self.y_min = y_min
        self.y_diff = y_max - y_min
        if num_layers is None:
            num_layers = 1
        if num_blocks is None:
            num_blocks = 2
        if num_blocks % 2 != 0:
            raise ValueError("`num_blocks` should be divided by 2")
        if latent_dim is None:
            latent_dim = 512
        if transition_builder is None:
            transition_builder = default_transition_builder
        # pseudo invertible q / y
        kwargs = {"num_layers": num_layers, "condition_dim": in_dim}
        q_to_latent_builder = monotonous_builder(True, True, True, **kwargs)
        q_from_latent_builder = monotonous_builder(True, False, False, **kwargs)
        self.q_invertible = PseudoInvertibleBlock(
            1,
            latent_dim,
            to_transition_builder=q_to_latent_builder,
            from_transition_builder=q_from_latent_builder,
        )
        y_to_latent_builder = monotonous_builder(True, False, True, **kwargs)
        y_from_latent_builder = monotonous_builder(True, True, False, **kwargs)
        self.y_invertible = PseudoInvertibleBlock(
            1,
            latent_dim,
            to_transition_builder=y_to_latent_builder,
            from_transition_builder=y_from_latent_builder,
        )
        q_params1 = list(self.q_invertible.to_latent.parameters())
        q_params2 = list(self.y_invertible.from_latent.parameters())
        self.q_parameters = q_params1 + q_params2
        # invertible blocks
        self.num_blocks = num_blocks
        self.block_parameters: List[nn.Parameter] = []
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = InvertibleBlock(latent_dim, transition_builder=transition_builder)
            self.block_parameters.extend(block.parameters())
            self.blocks.append(block)

    @property
    def q_fn(self) -> Callable[[Tensor], Tensor]:
        return lambda q: 2.0 * q - 1.0

    @property
    def y_fn(self) -> Callable[[Tensor], Tensor]:
        return lambda y: (y - self.y_min) / (0.5 * self.y_diff) - 1.0

    @property
    def q_inv_fn(self) -> Callable[[Tensor], Tensor]:
        return torch.sigmoid

    @property
    def y_inv_fn(self) -> Callable[[Tensor], Tensor]:
        return lambda y: (y + 1.0) * (0.5 * self.y_diff) + self.y_min

    def _detach_q(self) -> context_error_handler:
        def switch(requires_grad: bool) -> None:
            switch_requires_grad(self.q_parameters, requires_grad)
            switch_requires_grad(self.block_parameters, requires_grad)

        class _(context_error_handler):
            def __enter__(self) -> None:
                switch(False)

            def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                switch(True)

        return _()

    def _q_results(
        self,
        net: Tensor,
        q_batch: Optional[Tensor] = None,
        auto_encode: bool = False,
        do_inverse: bool = False,
        median: bool = False,
    ) -> Dict[str, Optional[Tensor]]:
        # prepare q_latent
        if q_batch is not None:
            q_batch = self.q_fn(q_batch)
        elif median:
            if q_batch is not None:
                msg = "`median` is specified but `q_batch` is still provided"
                raise ValueError(msg)
            q_batch = net.new_zeros(len(net), 1)
        if q_batch is None:
            q1 = q2 = q_latent = None
        else:
            q1, q2 = self.q_invertible(q_batch, net)
            q_latent = torch.cat([q1, q2], dim=1)
        # simulate quantile function
        q_ae = q_inverse = None
        y_inverse_latent = yq_inverse_latent = None
        if q_latent is None:
            y = qy_latent = None
        else:
            assert q1 is not None and q2 is not None
            if auto_encode:
                q_ae_logit = self.q_invertible.inverse((q1, q2), net)
                q_ae = self.q_inv_fn(q_ae_logit)
            for block in self.blocks:
                q1, q2 = block(q1, q2)
            qy_latent = torch.cat([q1, q2], dim=1)
            y = self.y_invertible.inverse((q1, q2), net)
            y = self.y_inv_fn(y)
            if do_inverse:
                inverse_results = self.forward(
                    net,
                    y_batch=y.detach(),
                    do_inverse=False,
                )
                q_inverse = inverse_results["q"]
                y_inverse_latent = inverse_results["y_latent"]
                yq_inverse_latent = inverse_results["yq_latent"]
        return {
            "y": y,
            "q_ae": q_ae,
            "q_latent": q_latent,
            "qy_latent": qy_latent,
            "q_inverse": q_inverse,
            "y_inverse_latent": y_inverse_latent,
            "yq_inverse_latent": yq_inverse_latent,
        }

    def _y_results(
        self,
        net: Tensor,
        y_batch: Optional[Tensor] = None,
        auto_encode: bool = False,
        do_inverse: bool = False,
    ) -> Dict[str, Optional[Tensor]]:
        # prepare y_latent
        if y_batch is None:
            y1 = y2 = y_latent = None
        else:
            y_batch = self.y_fn(y_batch)
            y1, y2 = self.y_invertible(y_batch, net)
            y_latent = torch.cat([y1, y2], dim=1)
        # simulate cdf
        y_ae = y_inverse = None
        q_inverse_latent = qy_inverse_latent = None
        if y_latent is None:
            q = q_logit = yq_latent = None
        else:
            if auto_encode:
                y_ae = self.y_invertible.inverse((y1, y2), net)
                y_ae = self.y_inv_fn(y_ae)
            for i in range(self.num_blocks):
                y1, y2 = self.blocks[self.num_blocks - i - 1].inverse(y1, y2)
            yq_latent = torch.cat([y1, y2], dim=1)
            q_logit = self.q_invertible.inverse((y1, y2), net)
            q = self.q_inv_fn(q_logit)
            with self._detach_q():
                if not do_inverse:
                    q_inverse_latent = self.q_invertible(q.detach(), net)
                    q_inverse_latent = torch.cat(q_inverse_latent, dim=1)
                else:
                    inverse_results = self.forward(
                        net,
                        q_batch=q,
                        do_inverse=False,
                    )
                    y_inverse = inverse_results["y"]
                    q_inverse_latent = inverse_results["q_latent"]
                    qy_inverse_latent = inverse_results["qy_latent"]
        return {
            "q": q,
            "q_logit": q_logit,
            "y_ae": y_ae,
            "y_latent": y_latent,
            "yq_latent": yq_latent,
            "y_inverse": y_inverse,
            "q_inverse_latent": q_inverse_latent,
            "qy_inverse_latent": qy_inverse_latent,
        }

    def forward(
        self,
        net: Tensor,
        *,
        q_batch: Optional[Tensor] = None,
        y_batch: Optional[Tensor] = None,
        auto_encode: bool = False,
        do_inverse: bool = False,
        median: bool = False,
    ) -> Dict[str, Optional[Tensor]]:
        results: Dict[str, Optional[Tensor]] = {}
        results.update(self._q_results(net, q_batch, auto_encode, do_inverse, median))
        results.update(self._y_results(net, y_batch, auto_encode, do_inverse))
        return results


__all__ = ["DDRCore"]
