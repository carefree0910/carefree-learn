import torch

from torch import nn
from typing import Optional
from typing import NamedTuple

from .discriminators import DiscriminatorOutput
from ....misc.toolkit import get_gradient


class GANTarget(NamedTuple):
    is_real: bool
    labels: Optional[torch.Tensor] = None


class GradientNormLoss(nn.Module):
    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def forward(self, net_input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        gradients = get_gradient(output, net_input, True, True)
        gradients = gradients.view(net_input.shape[0], -1)  # type: ignore
        gradients_norm = gradients.norm(2, dim=1)
        return torch.mean((gradients_norm - self.k) ** 2)


class GANLoss(nn.Module):
    def __init__(self, gan_mode: str):
        super().__init__()
        self.loss: nn.Module
        self.gan_mode = gan_mode
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = GradientNormLoss(k=1.0)
        else:
            raise NotImplementedError(f"gan mode {gan_mode} not implemented")
        self.ce = nn.CrossEntropyLoss()

    def expand_target(self, tensor: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = self.real_label if is_real else self.fake_label
        return target.expand_as(tensor)  # type: ignore

    def forward(self, output: DiscriminatorOutput, target: GANTarget) -> torch.Tensor:
        prediction, is_real = output.output, target.is_real
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.expand_target(prediction, is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            loss = -prediction.mean() if is_real else prediction.mean()
        else:
            raise NotImplementedError(f"gan_mode '{self.gan_mode}' is not implemented")
        if output.cond_logits is not None and target.is_real:
            cond_loss = self.ce(output.cond_logits, target.labels)
            loss = loss + cond_loss
        return loss


__all__ = [
    "GANTarget",
    "GradientNormLoss",
    "GANLoss",
]
