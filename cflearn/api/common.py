import torch

from typing import Any
from typing import Dict
from typing import Optional
from typing import Protocol
from pathlib import Path
from torch.nn import Module
from cftool.types import tensor_dict_type
from cftool.data_structures import Pool
from cftool.data_structures import IPoolItem
from cftool.data_structures import PoolItemContext
from torch.cuda.amp.autocast_mode import autocast

from ..schema import device_type
from ..toolkit import is_cpu
from ..toolkit import to_eval
from ..toolkit import empty_cuda_cache
from ..toolkit import get_torch_device
from ..parameters import OPT


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
        force_not_lazy: bool = False,
    ):
        self.m = to_eval(m)
        self.to(device, use_amp=use_amp, use_half=use_half)
        self.force_not_lazy = force_not_lazy

    @property
    def dtype(self) -> torch.dtype:
        return torch.float16 if self.use_half else torch.float32

    @property
    def to_half(self) -> bool:
        return self.use_amp or self.use_half

    @property
    def amp_context(self) -> autocast:
        return autocast(enabled=self.use_amp)

    def setup(self, device: device_type, use_amp: bool, use_half: bool) -> bool:
        """Returns whether anything has changed."""

        if use_amp and use_half:
            raise ValueError("`use_amp` & `use_half` should not be True simultaneously")
        new_device = get_torch_device(device)
        if (
            all(hasattr(self, k) for k in ["device", "use_amp", "use_half"])
            and self.device == new_device
            and self.use_amp == use_amp
            and self.use_half == use_half
        ):
            return False
        self.device = new_device
        self.use_amp = use_amp
        self.use_half = use_half
        return True

    def to(
        self,
        device: device_type,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> None:
        if not self.setup(device, use_amp, use_half):
            return
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

    @property
    def lazy_load(self) -> bool:
        return OPT.lazy_load_api and not self.force_not_lazy

    @property
    def need_change_device(self) -> bool:
        return self.lazy_load and not OPT.use_cpu_api and torch.cuda.is_available()

    def load(self, *, no_change: bool = False, **kwargs: Any) -> None:
        if not no_change and self.need_change_device:
            kwargs.setdefault("device", "cuda:0")
            self.to(**kwargs)

    def unload(self) -> None:
        if self.need_change_device:
            self.collect()

    def collect(self) -> None:
        device = self.device
        if not is_cpu(device):
            self.to("cpu")
        empty_cuda_cache(device)


class Weights(Pool[tensor_dict_type]):
    def register(self, key: str, path: Path) -> None:  # type: ignore
        super().register(key, lambda: torch.load(path))


class APIInitializer(Protocol):
    def __call__(
        self,
        *,
        device: device_type,
        use_half: bool,
        force_not_lazy: bool = False,
    ) -> IAPI:
        pass


class APIPool(Pool[IAPI]):
    def __init__(self, limit: int = -1, *, allow_duplicate: bool = True):
        super().__init__(limit, allow_duplicate=allow_duplicate)
        self.custom_use_halfs: Dict[str, bool] = {}

    def register(
        self,
        key: str,
        initializer: APIInitializer,
        *,
        use_half: Optional[bool] = None,
        force_not_lazy: bool = False,
        **kwargs: Any,
    ) -> None:
        if use_half is not None:
            self.custom_use_halfs[key] = use_half
        if (
            OPT.use_cpu_api
            or not torch.cuda.is_available()
            or (OPT.lazy_load_api and not force_not_lazy)
        ):
            device = "cpu"
            if use_half is None:
                use_half = False
        else:
            device = "cuda:0"
            if use_half is None:
                use_half = True
        init_fn = lambda: initializer(
            device=device,
            use_half=use_half,
            force_not_lazy=force_not_lazy,
        )
        super().register(key, init_fn, **kwargs)

    def use(self, key: str, **kwargs: Any) -> PoolItemContext:
        kwargs.setdefault("use_half", self.custom_use_halfs.get(key, True))
        return super().use(key, **kwargs)


__all__ = [
    "IAPI",
    "Weights",
    "APIPool",
]
