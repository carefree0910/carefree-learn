import torch

from torch import Tensor
from typing import Optional


def extract_to(array: Tensor, indices: Tensor, num_dim: int) -> Tensor:
    b = indices.shape[0]
    out = array.gather(-1, indices).contiguous()
    return out.view(b, *([1] * (num_dim - 1)))


def get_timesteps(t: int, num: int, device: torch.device) -> Tensor:
    return torch.full((num,), t, device=device, dtype=torch.long)


def q_sample(
    net: Tensor,
    timesteps: Tensor,
    sqrt_alphas_cumprod: Tensor,
    sqrt_one_minus_alphas_cumprod: Tensor,
    noise: Optional[Tensor] = None,
) -> Tensor:
    num_dim = len(net.shape)
    w_net = extract_to(sqrt_alphas_cumprod, timesteps, num_dim)
    w_noise = extract_to(sqrt_one_minus_alphas_cumprod, timesteps, num_dim)
    if noise is None:
        noise = torch.randn_like(net)
    net = w_net * net + w_noise * noise
    return net
