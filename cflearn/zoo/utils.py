import json

from typing import Any
from typing import Dict
from typing import NamedTuple
from pathlib import Path
from cftool.misc import update_dict
from cftool.misc import print_warning

from ..constants import DEFAULT_ZOO_TAG


root = Path(__file__).parent
configs_root = root / "configs"


class ParsedInfo(NamedTuple):
    json_path: Path
    download_name: str


def parse_config_info(config: str) -> ParsedInfo:
    tag = DEFAULT_ZOO_TAG
    module_type, module_name = config.split("/")
    download_name = module_name
    if "." in module_name:
        module_name, tag = module_name.split(".")
    json_folder = configs_root / module_type / module_name
    json_path = json_folder / f"{tag}.json"
    if not json_path.is_file():
        print_warning(f"cannot find '{tag}', fallback to '{DEFAULT_ZOO_TAG}'")
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
