import torch

from torch import Tensor
from typing import Union
from cftool.array import tensor_dict_type


cond_type = Union[Tensor, tensor_dict_type]
ADM_KEY = "labels"
ADM_TYPE = "adm"
CONCAT_KEY = "concat"
CONCAT_TYPE = "concat"
HYBRID_TYPE = "hybrid"
CROSS_ATTN_KEY = "context"
CROSS_ATTN_TYPE = "cross_attn"
CONTROL_HINT_KEY = "hint"
CONTROL_HINT_START_KEY = "hint_start"


def extract_to(array: Tensor, indices: Tensor, num_dim: int) -> Tensor:
    b = indices.shape[0]
    out = array.gather(-1, indices).contiguous()
    return out.view(b, *([1] * (num_dim - 1)))


def get_timesteps(t: int, num: int, device: torch.device) -> Tensor:
    return torch.full((num,), t, device=device, dtype=torch.long)
