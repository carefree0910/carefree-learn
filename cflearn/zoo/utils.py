import json

from typing import Any
from typing import Dict
from typing import NamedTuple
from pathlib import Path
from cftool.misc import update_dict

from ..constants import DEFAULT_ZOO_TAG


root = Path(__file__).parent
configs_root = root / "configs"


class ParsedInfo(NamedTuple):
    json_path: Path
    download_name: str


def parse_config_info(config: str) -> ParsedInfo:
    tag = DEFAULT_ZOO_TAG
    model_type, model_name = config.split("/")
    download_name = model_name
    if "." in model_name:
        model_name, tag = model_name.split(".")
    json_folder = configs_root / model_type / model_name
    json_path = json_folder / f"{tag}.json"
    if not json_path.is_file():
        json_path = json_folder / f"{DEFAULT_ZOO_TAG}.json"
    return ParsedInfo(json_path, download_name)


def parse_json(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r") as f:
        config = json.load(f)
    bases = config.pop("__bases__", [])
    for base in bases[::-1]:
        parsed = parse_config_info(base)
        config = update_dict(config, parse_json(parsed.json_path))
    return config


def parse_config(config: str) -> Dict[str, Any]:
    return parse_json(parse_config_info(config).json_path)


__all__ = [
    "parse_config_info",
    "parse_json",
    "parse_config",
]
