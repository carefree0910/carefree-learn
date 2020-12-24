import torch

from typing import Any
from typing import Dict
from typing import Optional

from ..base import HeadBase
from ....modules.blocks import DNDF
from ....modules.blocks import Linear


@HeadBase.register("dndf")
class DNDFHead(HeadBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dndf_config: Optional[Dict[str, Any]],
    ):
        super().__init__(in_dim, out_dim)
        self.dndf: Optional[DNDF]
        self.linear: Optional[torch.nn.Module]
        if dndf_config is not None:
            self.dndf = DNDF(in_dim, out_dim, **dndf_config)
            self.linear = None
        else:
            self.dndf = None
            self.linear = Linear(in_dim, out_dim)
            self.log_msg(  # type: ignore
                "`config` is not provided for `DNDFHead`, "
                "a `Linear` will be used instead of `DNDF`",
                self.warning_prefix,  # type: ignore
            )

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        if self.dndf is None:
            assert self.linear is not None
            return self.linear(net)
        return self.dndf(net)


__all__ = ["DNDFHead"]
