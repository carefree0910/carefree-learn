import torch

from typing import Any
from typing import Dict
from typing import Optional
from pathlib import Path
from torch.nn import Module

from .utils import parse_config
from .utils import parse_config_info
from ..toolkit import download_checkpoint
from ..modules.common import build_module


def build_predefined_module(config: str, **kwargs: Any) -> Module:
    parsed = parse_config(config)
    model_name = parsed.get("model_name")
    if model_name is None:
        raise ValueError(f"model name not found in '{parsed}'")
    model_config = parsed.get("model_config", {})
    return build_module(model_name, config=model_config, **kwargs)


def load_pretrained_weights(
    module: Module,
    tag: str,
    *,
    download_root: Optional[Path] = None,
    download_kwargs: Optional[Dict[str, Any]] = None,
) -> Module:
    ckpt_path = download_checkpoint(tag, download_root, **(download_kwargs or {}))
    module.load_state_dict(torch.load(ckpt_path))
    return module


def load_pretrained_module(
    config: str,
    *,
    tag: Optional[str] = None,
    download_root: Optional[Path] = None,
    download_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Module:
    module = build_predefined_module(config, **kwargs)
    load_pretrained_weights(
        module,
        tag=tag or parse_config_info(config).download_name,
        download_root=download_root,
        download_kwargs=download_kwargs,
    )
    return module


def load_module(
    config: str,
    *,
    pretrained: bool = False,
    tag: Optional[str] = None,
    download_root: Optional[Path] = None,
    download_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Module:
    if not pretrained:
        return build_predefined_module(config, **kwargs)
    return load_pretrained_module(
        config,
        tag=tag,
        download_root=download_root,
        download_kwargs=download_kwargs,
        **kwargs,
    )


__all__ = [
    "build_predefined_module",
    "load_pretrained_weights",
    "load_pretrained_module",
    "load_module",
]
