from typing import Dict
from typing import List
from typing import Callable
from typing import NamedTuple
from cftool.misc import print_warning


class BackboneInfo(NamedTuple):
    fn: Callable
    out_channels: List[int]
    return_nodes: Dict[str, str]


backbone_info_dict: Dict[str, BackboneInfo] = {}


def register_backbone(
    name: str,
    out_channels: List[int],
    return_nodes: Dict[str, str],
) -> Callable:
    def _register(fn: Callable) -> Callable:
        registered = backbone_info_dict.get(name)
        if registered is not None:
            print_warning(
                f"'{name}' has already been registered "
                f"in the given global dict ({backbone_info_dict})"
            )
            return fn
        backbone_info_dict[name] = BackboneInfo(fn, out_channels, return_nodes)
        return fn

    return _register


__all__ = [
    "register_backbone",
    "backbone_info_dict",
]
