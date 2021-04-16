import torch

from typing import *

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


__all__ = ["optimizer_dict", "register_optimizer"]
