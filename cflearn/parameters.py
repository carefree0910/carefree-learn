from pathlib import Path
from typing import Any
from typing import Dict
from cftool.misc import OPTBase


class OPTClass(OPTBase):
    cache_dir: Path
    external_dir: Path
    meta_settings: Dict[str, Any]

    @property
    def env_key(self) -> str:
        return "CFLEARN_ENV"

    @property
    def defaults(self) -> Dict[str, Any]:
        user_dir = Path.home()
        return dict(
            cache_dir=user_dir / ".cache" / "carefree-learn",
            external_dir=user_dir / ".cache" / "carefree-learn" / "external",
            meta_settings={},
        )

    @property
    def data_cache_dir(self) -> Path:
        return self.cache_dir / "data"

    def update_from_env(self) -> None:
        super().update_from_env()
        self._opt["cache_dir"] = Path(self._opt["cache_dir"])
        self._opt["external_dir"] = Path(self._opt["external_dir"])


OPT = OPTClass()


__all__ = [
    "OPT",
]
