from torch import nn
from torch import Tensor
from typing import Any
from typing import List
from typing import Tuple
from torch.nn import Module


TQKV = Tuple[Tensor, Tensor, Tensor]


class IHook(Module):
    # modify the `input`
    def before_forward(self, inp: Any) -> Any:
        return inp

    # modify the `output`
    def after_forward(self, inp: Any, out: Any) -> Any:
        return out


class IBasicHook(IHook):
    def after_forward(self, inp: Tensor, out: Tensor) -> Tensor:
        pass


class IAttentionHook(IHook):
    def after_forward(self, qkv_inp: TQKV, qkv_out: TQKV) -> TQKV:
        pass


class MultiHooks(IHook):
    def __init__(self, hooks: List[IHook]) -> None:
        super().__init__()
        self.hooks = nn.ModuleList(hooks)

    def before_forward(self, inp: Any) -> Any:
        for hook in self.hooks:
            inp = hook.before_forward(inp)
        return inp

    def after_forward(self, inp: Any, out: Any) -> Any:
        for hook in self.hooks:
            out = hook.after_forward(inp, out)
        return out
