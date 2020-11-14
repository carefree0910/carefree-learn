from torch import Tensor
from typing import Any
from typing import Dict
from typing import Union
from typing import Optional
from typing import NamedTuple
from torch.nn import Module

from .heads import HeadBase
from .transform import Transform
from .transform import SplitFeatures
from .extractors import ExtractorBase


class PipeConfig(NamedTuple):
    transform: str
    extractor: str
    head: str
    extractor_config: str
    head_config: str
    extractor_meta_scope: Optional[str]
    head_meta_scope: Optional[str]

    @property
    def use_extractor_meta(self) -> bool:
        return self.extractor_meta_scope is not None

    @property
    def use_head_meta(self) -> bool:
        return self.head_meta_scope is not None

    @property
    def extractor_scope(self) -> str:
        if self.extractor_meta_scope is None:
            return self.extractor
        return self.extractor_meta_scope

    @property
    def head_scope(self) -> str:
        if self.head_meta_scope is None:
            return self.head
        return self.head_meta_scope

    @property
    def extractor_key(self) -> str:
        if self.extractor_meta_scope is None:
            prefix = self.extractor
        else:
            prefix = self.extractor_meta_scope
        return f"{prefix}_{self.extractor_config}"

    @property
    def head_key(self) -> str:
        if self.head_meta_scope is None:
            prefix = self.head
        else:
            prefix = self.head_meta_scope
        return f"{prefix}_{self.head_config}"


class Pipe(Module):
    def __init__(self, transform: Transform, extractor: ExtractorBase, head: HeadBase):
        super().__init__()
        self.transform = transform
        self.extractor = extractor
        self.head = head

    def forward(
        self,
        inp: Union[Tensor, SplitFeatures],
        extract_kwargs: Optional[Dict[str, Any]] = None,
        head_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        net = inp if isinstance(inp, Tensor) else self.transform(inp)
        if extract_kwargs is None:
            extract_kwargs = {}
        net = self.extractor(net, **extract_kwargs)
        net_shape = net.shape
        if self.extractor.flatten_ts:
            if len(net_shape) == 3:
                net = net.view(net_shape[0], -1)
        if head_kwargs is None:
            head_kwargs = {}
        return self.head(net, **head_kwargs)


class PipePlaceholder(Pipe):
    def forward(
        self,
        inp: Union[Tensor, SplitFeatures],
        extract_kwargs: Optional[Dict[str, Any]] = None,
        head_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise ValueError("`PipePlaceholder.forward` should not be called")


__all__ = ["Pipe", "PipeConfig", "PipePlaceholder"]
