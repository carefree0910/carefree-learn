import gc
import time
import torch

import torch.nn as nn

from typing import Any
from typing import Dict
from typing import Type
from typing import TypeVar
from typing import Optional
from cftool.misc import TIME_FORMAT
from cftool.misc import print_info
from cftool.misc import sort_dict_by_value
from cftool.types import tensor_dict_type
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


class Weights:
    _weights: Optional[tensor_dict_type]

    def __init__(self, path: str, *, to_disk: bool = False):
        self.path = path
        self.to_disk = to_disk
        self.load_time = time.time()
        self._weights = None if to_disk else torch.load(path)

    def load(self) -> tensor_dict_type:
        self.load_time = time.time()
        if self._weights is None:
            self._weights = torch.load(self.path)
        return self._weights

    def unload(self) -> None:
        self._weights = None
        gc.collect()


class WeightsPool:
    pool: Dict[str, Weights]
    activated: Dict[str, Weights]

    # set `limit` to negative values to indicate 'no limit'
    def __init__(self, limit: int = -1):
        self.pool = {}
        self.activated = {}
        self.limit = limit
        if limit == 0:
            raise ValueError(
                "limit should either be negative "
                "(which indicates 'no limit') or be positive"
            )

    def __contains__(self, key: str) -> bool:
        return key in self.pool

    def register(self, key: str, path: str) -> None:
        if key in self.pool:
            raise ValueError(f"key '{key}' already exists")
        to_disk = self.limit > 0
        w = Weights(path, to_disk=to_disk)
        self.pool[key] = w
        if not to_disk:
            self.activated[key] = w
        else:
            if len(self.activated) < self.limit:
                w.load()
                self.activated[key] = w

    def get(self, key: str) -> tensor_dict_type:
        w = self.pool.get(key)
        if w is None:
            raise ValueError(f"key '{key}' does not exist")
        d = w.load()
        if key in self.activated:
            return d
        load_times = {k: v.load_time for k, v in self.activated.items()}
        earliest_key = list(sort_dict_by_value(load_times).keys())[0]
        self.activated.pop(earliest_key).unload()
        self.activated[key] = w
        print_info(
            f"'{earliest_key}' is unloaded to make room for '{key}' "
            f"(last updated: {time.strftime(TIME_FORMAT, time.localtime(w.load_time))})"
        )
        return d
