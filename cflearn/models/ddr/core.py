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
        def latent_builder() -> nn.Module:
            if to_transition_builder is not None:
                return to_transition_builder(in_dim, latent_dim)
            return MLP.simple(in_dim, None, [latent_dim, latent_dim], activation="mish")

        pseudo_builder = lambda: PseudoInvertibleBlock(
            1,
            latent_dim,
            to_transition_builder=to_transition_builder,
            from_transition_builder=from_transition_builder,
        )
        # to latent
        if latent_dim is None:
            latent_dim = 512
        self.to_latent = latent_builder()
        # pseudo invertible q / y
        self.q_invertible = pseudo_builder()
        self.y_invertible = pseudo_builder()
        q_params1 = list(self.q_invertible.to_latent.parameters())
        q_params2 = list(self.y_invertible.from_latent.parameters())
        self.q_parameters = q_params1 + q_params2
        # transition builder
        def default_transition_builder(dim: int) -> nn.Module:
            h_dim = int(dim // 2)
            return MLP.simple(h_dim, None, [h_dim, h_dim], activation="mish")

        if transition_builder is None:
            transition_builder = default_transition_builder
        # invertible blocks
        if num_blocks is None:
            num_blocks = 4
        if num_blocks % 2 != 0:
            raise ValueError("`num_blocks` should be divided by 2")
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()
        permutation_indices = torch.arange(latent_dim)
        for _ in range(num_blocks):
            block = InvertibleBlock(latent_dim, transition_builder)
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
        if q_latent is None:
            y = None
        else:
            q_net = latent if q_latent is None else latent + q_latent
            for block in self.blocks:
                q_net = block(q_net)
            permuted = (latent + q_latent)[..., self.permutation_indices]
            q_net = q_net - permuted
            y = self.y_invertible.inverse(q_net)
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
            permuted = (latent + y_latent)[..., self.permutation_indices]
            q = torch.tanh(self.q_invertible.inverse(y_net - permuted))
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
