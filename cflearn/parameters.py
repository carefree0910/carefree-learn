from typing import Any
from typing import Dict
from pathlib import Path
from cftool.misc import OPTBase


class OPTClass(OPTBase):
    cache_dir: Path
    external_dir: Path
    # this is for `run_multiple` (cflearn/api.py -> run_multiple)
    meta_settings: Dict[str, Any]
    # this is for extended usages
    # > e.g., if a library uses cflearn and wants to use this class to manage its settings
    external_settings: Dict[str, Any]
    # api settings
    use_cpu_api: bool
    lazy_load_api: bool
    ## sd api settings
    sd_weights_pool_limit: int

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
            external_settings={},
            use_cpu_api=False,
            lazy_load_api=False,
            sd_weights_pool_limit=-1,
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
