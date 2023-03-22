import torch

import torch.nn as nn

from typing import Any
from typing import Type
from typing import TypeVar
from typing import Optional
from torch.cuda.amp.autocast_mode import autocast

from .zoo import TPipeline
from ..misc.toolkit import is_cpu
from ..misc.toolkit import get_device
from ..misc.toolkit import empty_cuda_cache


TAPI = TypeVar("TAPI", bound="APIMixin")


class APIMixin:
    m: nn.Module
    device: torch.device
    use_amp: bool
    use_half: bool

    def __init__(
        self,
        m: nn.Module,
        device: torch.device,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ):
        if use_amp and use_half:
            raise ValueError("`use_amp` & `use_half` should not be True simultaneously")
        self.m = m
        self.device = device
        self.use_amp = use_amp
        self.use_half = use_half

    def to(
        self,
        device: torch.device,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> None:
        if use_amp and use_half:
            raise ValueError("`use_amp` & `use_half` should not be True simultaneously")
        self.device = device
        self.use_amp = use_amp
        self.use_half = use_half
        device_is_cpu = is_cpu(device)
        if device_is_cpu:
            self.m.to(device)
        if use_half:
            self.m.half()
        else:
            self.m.float()
        if not device_is_cpu:
            self.m.to(device)

    def empty_cuda_cache(self) -> None:
        empty_cuda_cache(self.device)

    @property
    def amp_context(self) -> autocast:
        return autocast(enabled=self.use_amp)

    @classmethod
    def from_pipeline(
        cls: Type[TAPI],
        m: TPipeline,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        **kwargs: Any,
    ) -> TAPI:
        if use_amp and use_half:
            raise ValueError("`use_amp` & `use_half` should not be True simultaneously")
        model = m.build_model.model
        if use_half:
            model.half()
        if device is not None:
            model.to(device)
        return cls(
            model,
            get_device(model),
            use_amp=use_amp,
            use_half=use_half,
            **kwargs,
        )
