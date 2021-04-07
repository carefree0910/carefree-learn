import torch

from typing import *
from torch.optim.optimizer import Optimizer

optimizer_dict = {}


def register_optimizer(name: str) -> Callable[[Type], Type]:
    def _register(cls_: Type) -> Type:
        global optimizer_dict
        optimizer_dict[name] = cls_
        return cls_

    return _register


register_optimizer("sgd")(torch.optim.SGD)
register_optimizer("adam")(torch.optim.Adam)
register_optimizer("adamw")(torch.optim.AdamW)
register_optimizer("rmsprop")(torch.optim.RMSprop)


@register_optimizer("nag")
class NAG(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, lr_old=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[Any]:
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            lr = group["lr"]
            lr_old = group.get("lr_old", lr)
            lr_correct = lr / lr_old
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = d_p.clone().zero_()
                buf = param_state["momentum_buffer"]
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(buf, alpha=momentum * momentum * lr_correct)
                p.data.add_(d_p, alpha=-(1 + momentum) * lr)
                buf.mul_(momentum * lr_correct).add_(d_p, alpha=-lr)
            group["lr_old"] = lr
        return loss


@register_optimizer("madgrad")
class MADGRAD(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        eps: float = 1.0e-6,
    ):
        defaults = dict(lr=lr, eps=eps, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        if "k" not in self.state:
            self.state["k"] = torch.tensor([0], dtype=torch.long)
        k = self.state["k"].item()

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"] + eps
            decay = group["weight_decay"]
            momentum = group["momentum"]

            ck = 1.0 - momentum
            lb = lr * math.sqrt(k + 1)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = "MADGRAD optimizer does not support sparse gradients"
                    raise RuntimeError(msg)

                state = self.state[p]
                if "grad_sum_sq" not in state:
                    state["grad_sum_sq"] = torch.zeros_like(p.data).detach()
                    state["s"] = torch.zeros_like(p.data).detach()
                    if momentum != 0.0:
                        state["x0"] = torch.clone(p.data).detach()

                grad_sum_sq = state["grad_sum_sq"]
                s = state["s"]

                if decay:
                    p.data.mul_(1.0 - lr * decay)

                if momentum == 0.0:
                    rms = grad_sum_sq.pow(1.0 / 3.0).add_(eps)
                    x0 = p.data.addcdiv(s, rms, value=1.0)
                else:
                    x0 = state["x0"]

                grad_sum_sq.addcmul_(grad, grad, value=lb)
                rms = grad_sum_sq.pow(1.0 / 3.0).add_(eps)

                s.data.add_(grad, alpha=lb)

                if momentum == 0.0:
                    p.data.copy_(x0.addcdiv(s, rms, value=-1.0))
                else:
                    z = x0.addcdiv(s, rms, value=-1.0)
                    p.data.mul_(1.0 - ck).add_(z, alpha=ck)

        self.state["k"] += 1
        return loss


__all__ = ["optimizer_dict", "register_optimizer"]
