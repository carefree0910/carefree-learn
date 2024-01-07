from typing import Any
from typing import Dict
from pathlib import Path
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


# meta settings


def meta_settings() -> Dict[str, Any]:
    return OPT.meta_settings


## api settings


def use_cpu_api() -> bool:
    return meta_settings().get("use_cpu_api", False)


def lazy_load_api() -> bool:
    return meta_settings().get("lazy_load_api", False)


__all__ = [
    "OPT",
    "meta_settings",
    "use_cpu_api",
    "lazy_load_api",
]
