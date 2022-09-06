import torch

from torch import Tensor


def extract_to(array: Tensor, indices: Tensor, num_dim: int) -> Tensor:
    b = indices.shape[0]
    out = array.gather(-1, indices).contiguous()
    return out.view(b, *([1] * (num_dim - 1)))


def get_timesteps(t: int, num: int, device: torch.device) -> Tensor:
    return torch.full((num,), t, device=device, dtype=torch.long)
