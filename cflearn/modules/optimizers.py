import math
import torch

import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Callable
from typing import Iterable
from typing import Optional
from torch.optim.optimizer import Optimizer


optimizer_dict: Dict[str, Type[Optimizer]] = {}


def register_optimizer(name: str) -> Callable:
    def _register(cls_: Type) -> Type:
        global optimizer_dict
        optimizer_dict[name] = cls_
        return cls_

    return _register


register_optimizer("sgd")(torch.optim.SGD)
register_optimizer("adam")(torch.optim.Adam)
register_optimizer("adamw")(torch.optim.AdamW)
register_optimizer("rmsprop")(torch.optim.RMSprop)


# AdamP
# reference : https://github.com/clovaai/AdamP


def channel_view(net: Tensor) -> Tensor:
    return net.view(net.shape[0], -1)


def layer_view(net: Tensor) -> Tensor:
    return net.view(1, -1)


def cosine_similarity(x: Tensor, y: Tensor, eps: float, view_func: Callable) -> Tensor:
    x = view_func(x)
    y = view_func(y)
    return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()


def projection(
    p: Tensor,
    grad: Tensor,
    perturb: Tensor,
    delta: float,
    wd_ratio: float,
    eps: float,
) -> Tuple[Tensor, float]:
    wd = 1.0
    expand_size = [-1] + [1] * (len(p.shape) - 1)
    for view_func in [channel_view, layer_view]:
        cosine_sim = cosine_similarity(grad, p.data, eps, view_func)
        if cosine_sim.max() < delta / math.sqrt(view_func(p.data).shape[1]):
            p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
            perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
            wd = wd_ratio
            return perturb, wd
    return perturb, wd


@register_optimizer("adamp")
class AdamP(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1.0e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            delta=delta,
            wd_ratio=wd_ratio,
            nesterov=nesterov,
        )
        super(AdamP, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Any:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group["betas"]
                nesterov = group["nesterov"]

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                # Adam
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                bias_correction2 = math.sqrt(bias_correction2)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = (exp_avg_sq.sqrt() / bias_correction2).add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1.0
                if len(p.shape) > 1:
                    perturb, wd_ratio = projection(
                        p,
                        grad,
                        perturb,
                        group["delta"],
                        group["wd_ratio"],
                        group["eps"],
                    )

                # Weight decay
                if group["weight_decay"] > 0.0:
                    p.data.mul_(1.0 - group["lr"] * group["weight_decay"] * wd_ratio)

                # Step
                p.data.add_(perturb, alpha=-step_size)

        return loss


__all__ = ["optimizer_dict", "register_optimizer"]
