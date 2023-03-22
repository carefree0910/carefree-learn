import torch

from torch import Tensor


def append_dims(net: Tensor, ndim: int) -> Tensor:
    diff = ndim - net.ndim
    return net[(...,) + (None,) * diff]


def append_zero(net: Tensor) -> Tensor:
    return torch.cat([net, net.new_zeros([1])])


# credit : https://github.com/LuChengTHU/dpm-solver/blob/88a41b53b2d5d7c5a665bcb55efda5f5a6061223/dpm_solver_pytorch.py#L1105
def interpolate_fn(x: Tensor, xp: Tensor, yp: Tensor) -> Tensor:
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    end_idx = torch.where(
        torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1
    )
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(
        y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)
    ).squeeze(2)
    end_y = torch.gather(
        y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)
    ).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand
