import torch

from typing import Any
from pathlib import Path
from torch.nn import Module
from cftool.types import tensor_dict_type
from cftool.data_structures import Pool
from cftool.data_structures import IPoolItem
from torch.cuda.amp.autocast_mode import autocast

from ..schema import device_type
from ..toolkit import is_cpu
from ..toolkit import to_eval
from ..toolkit import empty_cuda_cache
from ..toolkit import get_torch_device


class IAPI(IPoolItem):
    m: Module
    device: torch.device
    use_amp: bool
    use_half: bool

    def __init__(
        self,
        m: Module,
        device: device_type = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ):
        self.m = to_eval(m)
        self.to(device, use_amp=use_amp, use_half=use_half)

    @property
    def to_half(self) -> bool:
        return self.use_amp or self.use_half

    @property
    def amp_context(self) -> autocast:
        return autocast(enabled=self.use_amp)

    def setup(self, device: device_type, use_amp: bool, use_half: bool) -> None:
        if use_amp and use_half:
            raise ValueError("`use_amp` & `use_half` should not be True simultaneously")
        self.device = get_torch_device(device)
        self.use_amp = use_amp
        self.use_half = use_half

    def to(
        self,
        device: device_type,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> None:
        self.setup(device, use_amp, use_half)
        device_is_cpu = is_cpu(device)
        if device_is_cpu:
            self.m.to(device)
        if self.to_half:
            self.m.half()
        else:
            self.m.float()
        if not device_is_cpu:
            self.m.to(device)

    def empty_cuda_cache(self) -> None:
        empty_cuda_cache(self.device)

    # pool managements

    def load(self, *, device: device_type = None, **kwargs: Any) -> None:
        if device is not None and torch.cuda.is_available():
            kwargs.setdefault("use_half", True)
            self.to(device, **kwargs)

    def unload(self) -> None:
        device = self.device
        if not is_cpu(device):
            self.to("cpu")
        empty_cuda_cache(device)


class Weights(Pool[tensor_dict_type]):
    def register(self, key: str, path: Path) -> None:  # type: ignore
        super().register(key, lambda: torch.load(path))


__all__ = [
    "IAPI",
    "Weights",
]
