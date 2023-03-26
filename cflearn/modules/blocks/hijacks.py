import torch

from torch import nn
from torch import Tensor
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import NamedTuple
from cftool.misc import shallow_copy_dict
from cftool.array import tensor_dict_type

from .hooks import TQKV
from .hooks import IHook
from .hooks import IBasicHook
from .hooks import IAttentionHook
from .hooks import MultiHooks
from .customs import Linear


class IHijackMixin:
    hook: Optional[IHook]


# linear/conv hijacks


class IBasicHijackMixin(IHijackMixin):
    weight: Tensor

    def __init__(self, *args: Any, hook: Optional[IBasicHook] = None, **kwargs: Any):
        self.args = args
        self.kwargs = shallow_copy_dict(kwargs)
        super().__init__(*args, **kwargs)
        self.hook = hook

    def forward(self, net: Tensor) -> Tensor:
        inp = net
        net = super().forward(net)  # type: ignore
        if self.hook is not None:
            net = self.hook.callback(inp, net)
        return net


class HijackLinear(IBasicHijackMixin, nn.Linear):
    pass


class HijackCustomLinear(IBasicHijackMixin, Linear):
    pass


class HijackConv1d(IBasicHijackMixin, nn.Conv1d):
    pass


class HijackConv2d(IBasicHijackMixin, nn.Conv2d):
    pass


class HijackConv3d(IBasicHijackMixin, nn.Conv3d):
    pass


# attention hijacks


class IAttention:
    hook: Optional[IAttentionHook]
    input_dim: int
    embed_dim: int


# LoRA
## input -> lora_down -> selector -> lora_up -> dropout -> alpha (scale)


class ILoRAHook(IBasicHook):
    rank: int
    lora_down: Union[nn.Linear, nn.Conv2d]
    selector: Union[nn.Identity, nn.Linear, nn.Conv2d]
    lora_up: Union[nn.Linear, nn.Conv2d]
    dropout: nn.Dropout
    alpha: nn.Parameter

    @property
    def scale(self) -> float:
        return self.alpha.item() / self.lora_down.weight.shape[0]

    def set_scale(self, scale: float) -> None:
        self.alpha.data.fill_(scale * self.lora_down.weight.shape[0])

    def callback(self, inp: Tensor, out: Tensor) -> Tensor:
        lora = self.dropout(self.lora_up(self.selector(self.lora_down(inp))))
        return out + self.scale * lora

    @classmethod
    def create_with(cls, m: IBasicHijackMixin, rank: int) -> "ILoRAHook":
        self = cls(*m.args, rank=rank, **m.kwargs)
        self.to(m.weight)
        return self


class LoRALinearHook(ILoRAHook):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *args: Any,
        rank: int = 4,
        dropout: float = 0.1,
        scale: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than "
                f"{min(in_features, out_features)}"
            )
        self.rank = rank
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.selector = nn.Identity()
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_parameter("alpha", nn.Parameter(torch.tensor(scale)))

        nn.init.normal_(self.lora_down.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_up.weight)

    def set_selector(self, diag: Tensor) -> None:
        assert diag.shape == (self.rank,)
        self.selector = nn.Linear(self.rank, self.rank, bias=False)
        self.selector.weight.data = torch.diag(diag).to(self.lora_up.weight.data)


class LoRAConvHook(ILoRAHook):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        *args: Any,
        rank: int = 4,
        dropout: float = 0.1,
        scale: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if rank > min(in_channels, out_channels):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than "
                f"{min(in_channels, out_channels)}"
            )
        self.rank = rank
        self.lora_down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=rank,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.selector = nn.Identity()
        self.lora_up = nn.Conv2d(
            in_channels=rank,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.register_parameter("alpha", nn.Parameter(torch.tensor(scale)))

        nn.init.normal_(self.lora_down.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_up.weight)

    def set_selector(self, diag: Tensor) -> None:
        assert diag.shape == (self.rank,)
        self.selector = nn.Conv2d(
            in_channels=self.rank,
            out_channels=self.rank,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector.weight.data = torch.diag(diag).to(self.lora_up.weight.data)


class LoRAAttentionHook(IAttentionHook):
    def __init__(self, in_features: int, out_features: int, *, rank: int) -> None:
        super().__init__()
        self.to_q = LoRALinearHook(in_features, out_features, rank=rank)
        self.to_k = LoRALinearHook(in_features, out_features, rank=rank)
        self.to_v = LoRALinearHook(in_features, out_features, rank=rank)

    def set_scale(self, scale: float) -> None:
        self.to_q.set_scale(scale)
        self.to_k.set_scale(scale)
        self.to_v.set_scale(scale)

    def callback(self, qkv_inp: TQKV, qkv_out: TQKV) -> TQKV:
        q_in, k_in, v_in = qkv_inp
        q, k, v = qkv_out
        q = self.to_q.callback(q_in, q)
        k = self.to_k.callback(k_in, k)
        v = self.to_v.callback(v_in, v)
        return q, k, v

    @classmethod
    def create_with(cls, m: IAttention, rank: int) -> "LoRAAttentionHook":
        return cls(m.input_dim, m.embed_dim, rank=rank)


class LoRAPack(NamedTuple):
    rank: int
    hooks: Dict[str, Union[ILoRAHook, LoRAAttentionHook]]


class LoRAManager:
    def __init__(self) -> None:
        self.m = None
        self.injected = False
        self.lora_packs: Dict[str, LoRAPack] = {}

    # api

    def build_pack(self, *keys: str) -> Optional[LoRAPack]:
        if not keys:
            return None
        if len(keys) == 1:
            return self.lora_packs.get(keys[0])
        selected = [self.lora_packs.get(key) for key in keys]
        not_none = [pack for pack in selected if pack is not None]
        if not not_none:
            return None
        if len(not_none) == 1:
            return selected[0]
        selected_ranks = list(set(pack.rank for pack in not_none))
        rank = -1 if len(selected_ranks) > 1 else selected_ranks[0]
        hooks: Dict[str, List[IHook]] = {}
        for pack in not_none:
            for k, v in pack.hooks.items():
                hooks.setdefault(k, []).append(v)
        merged = {k: v[0] if len(v) == 1 else MultiHooks(v) for k, v in hooks.items()}
        return LoRAPack(rank, merged)

    def load_pack_with(self, key: str, d: tensor_dict_type) -> None:
        pack = self.lora_packs.get(key)
        if pack is None:
            raise ValueError(f"cannot find '{key}' pack")
        for k, v in pack.hooks.items():
            for n, p in v.named_parameters():
                tp = d[f"{k}.{n}"]
                assert p.data.shape == tp.shape
                p.data = tp.to(p.data)

    def prepare(
        self,
        m: nn.Module,
        *,
        key: str,
        rank: int,
        target_ancestors: Optional[Set[str]] = None,
    ) -> None:
        # cleanup existing hooks
        pack = self.lora_packs.pop(key, None)
        if pack is not None:
            for hook in pack.hooks.values():
                hook.cpu()
            torch.cuda.empty_cache()
        # prepare new hooks
        pack = LoRAPack(rank, {})
        self.m = m
        if target_ancestors is None:
            self._prepare(m, "", pack)
        else:
            for name, module in m.named_modules():
                if module.__class__.__name__ in target_ancestors:
                    self._prepare(module, name, pack)
        # set pack
        self.lora_packs[key] = pack

    def inject(self, m: nn.Module, *keys: str) -> None:
        if self.m is None:
            raise ValueError("LoRA is not prepared, please call `prepare` first.")
        if m is not self.m:
            raise ValueError("prepared module does not match the incoming module.")
        pack = self.build_pack(*keys)
        if pack is None:
            raise ValueError(f"cannot build LoRAPack with {', '.join(keys)}")
        hooks = pack.hooks
        pivot = list(m.parameters())[0]
        for name, module in m.named_modules():
            hook = hooks.get(name)
            if hook is not None:
                if not isinstance(module, IHijackMixin):
                    msg = f"`hook` is found for '{name}' but it is not a Hijack module"
                    raise ValueError(msg)
                hook.to(pivot)
                module.hook = hook
        self.injected = True

    def cleanup(self, m: nn.Module) -> None:
        if not self.injected:
            raise ValueError("LoRA is not injected, please call `inject` first.")
        if m is not self.m:
            raise ValueError("injected module does not match the incoming module.")
        for module in m.modules():  # type: ignore
            if isinstance(module, IHijackMixin):
                module.hook = None
        self.injected = False
        torch.cuda.empty_cache()

    def set_scale(self, key: str, scale: float) -> None:
        pack = self.lora_packs.get(key)
        if pack is None:
            raise ValueError(f"cannot find LoRAPack '{key}'")
        for hook in pack.hooks.values():
            hook.set_scale(scale)

    def set_scales(self, scales: Dict[str, float]) -> None:
        for k, v in scales.items():
            self.set_scale(k, v)

    def checkpoint(self, m: nn.Module) -> Dict[str, Optional[IHook]]:
        hooks: Dict[str, Optional[IHook]] = {}
        for key, module in m.named_modules():
            if isinstance(module, IHijackMixin):
                hooks[key] = module.hook
        return hooks

    def restore(self, m: nn.Module, hooks: Dict[str, Optional[IHook]]) -> None:
        if m is not self.m:
            raise ValueError("prepared module does not match the incoming module.")
        for key, module in m.named_modules():  # type: ignore
            if isinstance(module, IHijackMixin):
                module.hook = hooks[key]
        self.injected = True

    # internal

    def _prepare(self, m: nn.Module, prefix: str, pack: LoRAPack) -> None:
        rank = pack.rank
        for name, module in m.named_modules():
            key = f"{prefix}.{name}"
            if isinstance(module, (HijackLinear, HijackCustomLinear)):
                pack.hooks[key] = LoRALinearHook.create_with(module, rank)
            elif isinstance(module, HijackConv2d):
                pack.hooks[key] = LoRAConvHook.create_with(module, rank)
            elif isinstance(module, IAttention):
                pack.hooks[key] = LoRAAttentionHook.create_with(module, rank)
