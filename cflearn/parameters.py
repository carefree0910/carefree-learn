from typing import Any
from typing import Dict
from pathlib import Path
from cftool.misc import OPTBase


class OPTClass(OPTBase):
    cache_dir: Path
    external_dir: Path
    # this is used for `run_multiple` (cflearn/api.py -> run_multiple)
    meta_settings: Dict[str, Any]
    # api settings
    use_cpu_api: bool
    lazy_load_api: bool

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
            use_cpu_api=False,
            lazy_load_api=False,
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
