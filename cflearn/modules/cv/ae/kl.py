import torch

import numpy as np

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from cftool.types import tensor_dict_type

from .common import IAttentionAutoEncoder
from ..common import register_generator
from ..common import DecoderInputs
from ....constants import PREDICTIONS_KEY


class GaussianDistribution:
    key = "distribution"

    def __init__(self, net: Tensor, deterministic: bool = False):
        self.net = net
        self.device = net.device
        self.deterministic = deterministic
        self.mean, log_var = torch.chunk(net, 2, dim=1)
        self.log_var = torch.clamp(log_var, -30.0, 20.0)
        if self.deterministic:
            zeros = torch.zeros_like(self.mean)
            self.var = self.std = zeros
        else:
            self.std = torch.exp(0.5 * self.log_var)
            self.var = torch.exp(self.log_var)

    def sample(self) -> Tensor:
        std = self.std * torch.randn_like(self.mean)
        return self.mean + std

    def kl(self, other: Optional["GaussianDistribution"] = None) -> Tensor:
        if self.deterministic:
            return torch.tensor([0.0], device=self.device)
        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.log_var,
                dim=[1, 2, 3],
            )
        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.log_var
            + other.log_var,
            dim=[1, 2, 3],
        )

    def nll(self, sample: Tensor, dims: Tuple[int, ...] = (1, 2, 3)) -> Tensor:
        if self.deterministic:
            return torch.tensor([0.0], device=self.device)
        return 0.5 * torch.sum(
            np.log(2.0 * np.pi)
            + self.log_var
            + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> Tensor:
        return self.mean


@register_generator("ae_kl")
class AttentionAutoEncoderKL(IAttentionAutoEncoder):
    enc_double_channels = True

    def encode(self, net: Tensor) -> GaussianDistribution:
        net = self.generator.encoder.encode(net)
        net = self.to_embedding(net)
        return GaussianDistribution(net)

    def decode(self, inputs: DecoderInputs) -> Tensor:
        inputs.z = self.from_embedding(inputs.z)
        net = self.generator.decoder.decode(inputs)
        return net

    def get_results(
        self,
        net: Tensor,
        *,
        sample_posterior: bool = True,
        no_head: bool = False,
        apply_tanh: Optional[bool] = None,
    ) -> Tuple[Tensor, GaussianDistribution]:
        distribution = self.encode(net)
        z = distribution.sample() if sample_posterior else distribution.mode()
        net = self.decode(DecoderInputs(z=z, no_head=no_head, apply_tanh=apply_tanh))
        return net, distribution

    def reconstruct(
        self,
        net: Tensor,
        *,
        labels: Optional[Tensor] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tensor]:
        kwargs = kwargs or {}
        kwargs["sample_posterior"] = False
        return self.get_results(net, **kwargs)[0]

    def forward(
        self,
        net: Tensor,
        *,
        sample_posterior: bool = True,
        no_head: bool = False,
        apply_tanh: Optional[bool] = None,
    ) -> tensor_dict_type:
        net, distribution = self.get_results(
            net,
            sample_posterior=sample_posterior,
            no_head=no_head,
            apply_tanh=apply_tanh,
        )
        return {PREDICTIONS_KEY: net, GaussianDistribution.key: distribution}


__all__ = [
    "AttentionAutoEncoderKL",
]
