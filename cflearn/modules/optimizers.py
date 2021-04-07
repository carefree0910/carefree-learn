import math
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


@register_optimizer("ranger")
class Ranger(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        alpha: float = 0.5,
        k: int = 6,
        n_sma_threshold: float = 5.0,  # 4.0
        betas: Tuple[float, float] = (0.95, 0.999),  # (0.90, 0.999)
        eps: float = 1e-5,
        weight_decay: float = 0.0,
        use_gc: bool = True,
        gc_conv_only: bool = False,
    ):
        defaults = dict(
            lr=lr,
            alpha=alpha,
            k=k,
            n_sma_threshhold=n_sma_threshold,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            step_counter=0,
        )
        super().__init__(params, defaults)
        self.n_sma_threshold = n_sma_threshold
        self.alpha = alpha

        self.radam_buffer = [[None, None, None] for _ in range(10)]
        self.use_gc = use_gc
        self.gc_gradient_threshold = 3 if gc_conv_only else 1

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    msg = "Ranger optimizer does not support sparse gradients"
                    raise RuntimeError(msg)

                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) != 0:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)
                else:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                    state["slow_buffer"] = torch.empty_like(p.data)
                    state["slow_buffer"].copy_(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                if grad.dim() > self.gc_gradient_threshold:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                state["step"] += 1

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state["step"] == buffered[0]:
                    n_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    n_sma_max = 2.0 / (1.0 - beta2) - 1.0
                    n_sma = n_sma_max - 2.0 * state["step"] * beta2_t / (1.0 - beta2_t)
                    buffered[1] = n_sma
                    if n_sma <= self.n_sma_threshold:
                        step_size = 1.0 / (1.0 - beta1 ** state["step"])
                    else:
                        step_size = math.sqrt(
                            (1.0 - beta2_t) * (n_sma - 4.0)
                            / (n_sma_max - 4.0) * (n_sma - 2.0)
                            / n_sma * n_sma_max
                            / (n_sma_max - 2.0)
                        ) / (1.0 - beta1 ** state["step"])
                    buffered[2] = step_size

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32,
                        alpha=-group["weight_decay"] * group["lr"],
                    )

                if n_sma <= self.n_sma_threshold:
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group["lr"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group["lr"])

                p.data.copy_(p_data_fp32)

                if state["step"] % group["k"] == 0:
                    slow_p = state["slow_buffer"]
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    p.data.copy_(slow_p)

        return loss


__all__ = ["optimizer_dict", "register_optimizer"]
