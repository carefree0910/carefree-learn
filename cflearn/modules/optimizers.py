import torch
from torch.optim.optimizer import Optimizer, required

optimizer_dict = {}


def register_optimizer(name):
    def _register(cls_):
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
    def __init__(self, params, lr=required, momentum=0, weight_decay=0):
        defaults = dict(lr=lr, lr_old=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
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


__all__ = ["optimizer_dict", "register_optimizer"]
