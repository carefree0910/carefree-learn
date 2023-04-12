import gc
import time
import torch

import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Type
from typing import Generic
from typing import TypeVar
from typing import Callable
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
TItem = TypeVar("TItem")


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
        self.m = m.eval().requires_grad_(False)
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


class ILoadableItem(Generic[TItem]):
    _item: Optional[TItem]

    def __init__(self, init_fn: Callable[[], TItem], *, init: bool = False):
        self.init_fn = init_fn
        self.load_time = time.time()
        self._item = init_fn() if init else None

    def load(self, **kwargs: Any) -> TItem:
        self.load_time = time.time()
        if self._item is None:
            self._item = self.init_fn()
        return self._item

    def unload(self) -> None:
        self._item = None
        gc.collect()


class ILoadablePool(Generic[TItem], metaclass=ABCMeta):
    pool: Dict[str, ILoadableItem]
    activated: Dict[str, ILoadableItem]

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

    def register(self, key: str, init_fn: Callable[[bool], ILoadableItem]) -> None:
        if key in self.pool:
            raise ValueError(f"key '{key}' already exists")
        init = self.limit < 0 or len(self.activated) < self.limit
        loadable_item = init_fn(init)
        self.pool[key] = loadable_item
        if init:
            self.activated[key] = loadable_item

    def get(self, key: str, **kwargs: Any) -> TItem:
        loadable_item = self.pool.get(key)
        if loadable_item is None:
            raise ValueError(f"key '{key}' does not exist")
        item = loadable_item.load(**kwargs)
        if key in self.activated:
            return item
        load_times = {k: v.load_time for k, v in self.activated.items()}
        earliest_key = list(sort_dict_by_value(load_times).keys())[0]
        self.activated.pop(earliest_key).unload()
        self.activated[key] = loadable_item

        time_format = "-".join(TIME_FORMAT.split("-")[:-1])
        print_info(
            f"'{earliest_key}' is unloaded to make room for '{key}' "
            f"(last updated: {time.strftime(time_format, time.localtime(loadable_item.load_time))})"
        )
        return item


class Weights(ILoadableItem[tensor_dict_type]):
    def __init__(self, path: str, *, init: bool = False):
        super().__init__(lambda: torch.load(path), init=init)


class WeightsPool(ILoadablePool[tensor_dict_type]):
    def register(self, key: str, path: str) -> None:  # type: ignore
        super().register(key, lambda init: Weights(path, init=init))
