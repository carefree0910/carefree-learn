import os

from typing import Any
from typing import Dict
from cftool.misc import OPTBase


class OPTClass(OPTBase):
    cache_dir: str
    meta_settings: Dict[str, Any]

    @property
    def env_key(self) -> str:
        return "CFLEARN_ENV"

    @property
    def defaults(self) -> Dict[str, Any]:
        return dict(
            cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "carefree-learn"),
            meta_settings={},
        )

    @property
    def data_cache_dir(self) -> str:
        return os.path.join(self.cache_dir, "data")


OPT = OPTClass()


__all__ = [
    "OPT",
]
