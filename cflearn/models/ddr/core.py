import torch

import torch.nn as nn

from typing import Dict
from typing import Callable
from typing import Optional

from ...misc.toolkit import switch_requires_grad
from ...modules.blocks import MLP
from ...modules.blocks import InvertibleBlock
from ...modules.blocks import PseudoInvertibleBlock


class DDRCore(nn.Module):
    def __init__(
        self,
        in_dim: int,
        y_min: float,
        y_max: float,
        num_blocks: Optional[int] = None,
        latent_dim: Optional[int] = None,
        transition_builder: Callable[[int], nn.Module] = None,
        to_transition_builder: Callable[[int, int], nn.Module] = None,
        from_transition_builder: Callable[[int, int], nn.Module] = None,
    ):
        super().__init__()
        self.y_min = y_min
        self.y_diff = y_max - y_min
        # builders
        def latent_builder(activation: str) -> nn.Module:
            if to_transition_builder is not None:
                return to_transition_builder(in_dim, latent_dim)
            return MLP.simple(in_dim, None, [latent_dim], activation=activation)

        pseudo_builder = lambda to_activation, from_activation: PseudoInvertibleBlock(
            1,
            latent_dim,
            to_activation=to_activation,
            from_activation=from_activation,
            to_transition_builder=to_transition_builder,
            from_transition_builder=from_transition_builder,
        )
        # to latent
        if latent_dim is None:
            latent_dim = 512
        self.to_latent = latent_builder("mish")
        # pseudo invertible q / y
        self.q_invertible = pseudo_builder("mish", "mish")
        self.y_invertible = pseudo_builder("mish", "mish")
        q_params1 = list(self.q_invertible.to_latent.parameters())
        q_params2 = list(self.y_invertible.from_latent.parameters())
        self.q_parameters = q_params1 + q_params2
        # add / multiply
        self.q_add = pseudo_builder("mish", "mish")
        self.q_mul = pseudo_builder("mish", "mish")
        self.y_add = pseudo_builder("mish", "mish")
        self.y_mul = pseudo_builder("mish", "mish")
        self.xq_add, self.xq_mul = latent_builder("mish"), latent_builder("mish")
        self.xy_add, self.xy_mul = latent_builder("mish"), latent_builder("mish")
        # transition builder
        def default_transition_builder(dim: int) -> nn.Module:
            h_dim = int(dim // 2)
            return MLP.simple(h_dim, None, [h_dim], activation="mish")

        if transition_builder is None:
            transition_builder = default_transition_builder
        # invertible blocks
        if num_blocks is None:
            num_blocks = 2
        if num_blocks % 2 != 0:
            raise ValueError("`num_blocks` should be divided by 2")
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()
        permutation_indices = torch.arange(latent_dim)
        for _ in range(num_blocks):
            block = InvertibleBlock(latent_dim, transition_builder=transition_builder)
            permutation_indices = permutation_indices[block.indices]
            self.blocks.append(block)
        self.register_buffer("permutation_indices", permutation_indices)

    @property
    def q_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda q: 2.0 * q - 1.0

    @property
    def q_inv_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda q: 0.5 * (q + 1.0)

    @property
    def y_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda y: (y - self.y_min) / (0.5 * self.y_diff) - 1.0

    @property
    def y_inv_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda y: (y + 1.0) * (0.5 * self.y_diff) + self.y_min

    def _get_q_results(
        self,
        net: torch.Tensor,
        latent: torch.Tensor,
        q_batch: Optional[torch.Tensor] = None,
        do_inverse: bool = False,
        median: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # prepare q_latent
        q_latent = None
        if not median:
            if q_batch is not None:
                q_batch = self.q_fn(q_batch)
                q_latent = self.q_invertible(q_batch)
        elif q_batch is not None:
            msg = "`median` is specified but `q_batch` is still provided"
            raise ValueError(msg)
        # simulate quantile function
        q_inverse = None
        if q_latent is None and not median:
            y = None
        else:
            q_net = latent if q_latent is None else latent + q_latent
            for block in self.blocks:
                q_net = block(q_net)
            if q_latent is not None:
                permuted = q_latent[..., self.permutation_indices]
                q_net = q_net - permuted
            y = self.y_invertible.inverse(q_net)
            if not median:
                add_latent, mul_latent = self.xq_add(net), self.xq_mul(net)
                q_add_latent = self.q_add(q_batch) + add_latent
                q_mul_latent = self.q_mul(q_batch) * mul_latent
                y_add = self.q_add.inverse(q_add_latent)
                y_mul = self.q_mul.inverse(q_mul_latent)
                y = y + y_add + y_mul
            y = self.y_inv_fn(y)
            if do_inverse:
                q_inverse = self.forward(
                    net,
                    latent,
                    y_batch=y.detach(),
                    do_inverse=False,
                )["q"]
        return {"y": y, "q_inverse": q_inverse}

    def _get_y_results(
        self,
        net: torch.Tensor,
        latent: torch.Tensor,
        y_batch: Optional[torch.Tensor] = None,
        do_inverse: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # prepare y_latent
        if y_batch is None:
            y_latent = None
        else:
            y_batch = self.y_fn(y_batch)
            y_latent = self.y_invertible(y_batch)
        # simulate cdf
        y_inverse = None
        if y_latent is None:
            q = None
        else:
            y_net = latent + y_latent
            for i in range(self.num_blocks):
                y_net = self.blocks[self.num_blocks - i - 1].inverse(y_net)
            permuted = y_latent[..., self.permutation_indices]
            q = self.q_invertible.inverse(y_net - permuted)
            add_latent, mul_latent = self.xy_add(net), self.xy_mul(net)
            q_add = self.y_add.inverse(self.y_add(y_batch) + add_latent)
            q_mul = self.y_mul.inverse(self.y_mul(y_batch) * mul_latent)
            q = torch.tanh(q + q_add + q_mul)
            q = self.q_inv_fn(q)
            if do_inverse:
                switch_requires_grad(self.q_parameters, False)
                y_inverse = self.forward(
                    net,
                    latent,
                    q_batch=q,
                    do_inverse=False,
                )["y"]
                switch_requires_grad(self.q_parameters, True)
        return {"q": q, "y_inverse": y_inverse}

    def forward(
        self,
        net: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
        *,
        q_batch: Optional[torch.Tensor] = None,
        y_batch: Optional[torch.Tensor] = None,
        do_inverse: bool = False,
        median: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if latent is None:
            latent = self.to_latent(net)
        results = {}
        results.update(self._get_q_results(net, latent, q_batch, do_inverse, median))
        results.update(self._get_y_results(net, latent, y_batch, do_inverse))
        return results


__all__ = ["DDRCore"]
