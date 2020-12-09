import os
import time
import shutil

from typing import Dict
from typing import List
from typing import Optional
from cftool.misc import LoggingMixin

SAVING_DELIM = "^_^"


def _to_saving_path(identifier: str, saving_folder: Optional[str]) -> str:
    if saving_folder is None:
        saving_path = identifier
    else:
        saving_path = os.path.join(saving_folder, identifier)
    return saving_path


def _make_saving_path(
    i: int,
    name: str,
    saving_path: str,
    remove_existing: bool,
) -> str:
    saving_path = os.path.abspath(saving_path)
    saving_folder, identifier = os.path.split(saving_path)
    postfix = f"{SAVING_DELIM}{name}{SAVING_DELIM}{i:04d}"
    if os.path.isdir(saving_folder) and remove_existing:
        for existing_model in os.listdir(saving_folder):
            if os.path.isdir(os.path.join(saving_folder, existing_model)):
                continue
            if existing_model.startswith(f"{identifier}{postfix}"):
                print(
                    f"{LoggingMixin.warning_prefix}"
                    f"'{existing_model}' was found, it will be removed"
                )
                os.remove(os.path.join(saving_folder, existing_model))
    return f"{saving_path}{postfix}"


def _fetch_saving_paths(
    identifier: str = "cflearn",
    saving_folder: Optional[str] = None,
) -> Dict[str, List[str]]:
    paths: Dict[str, List[str]] = {}
    saving_path = _to_saving_path(identifier, saving_folder)
    saving_path = os.path.abspath(saving_path)
    base_folder = os.path.dirname(saving_path)
    for existing_model in os.listdir(base_folder):
        if not os.path.isfile(os.path.join(base_folder, existing_model)):
            continue
        existing_model, existing_extension = os.path.splitext(existing_model)
        if existing_extension != ".zip":
            continue
        if SAVING_DELIM in existing_model:
            *folder, name, i = existing_model.split(SAVING_DELIM)
            if os.path.join(base_folder, SAVING_DELIM.join(folder)) != saving_path:
                continue
            new_path = _make_saving_path(int(i), name, saving_path, False)
            paths.setdefault(name, []).append(new_path)
    return paths


def _remove(identifier: str = "cflearn", saving_folder: str = None) -> None:
    for path_list in _fetch_saving_paths(identifier, saving_folder).values():
        for path in path_list:
            path = f"{path}.zip"
            print(f"{LoggingMixin.info_prefix}removing {path}...")
            os.remove(path)


def _rmtree(folder: str, patience: float = 10.0) -> None:
    if not os.path.isdir(folder):
        return None
    t = time.time()
    while True:
        try:
            if time.time() - t >= patience:
                prefix = LoggingMixin.warning_prefix
                print(f"\n{prefix}failed to rmtree: {folder}")
                break
            shutil.rmtree(folder)
            break
        except:
            print("", end=".", flush=True)
            time.sleep(1)


__all__ = [
    "_to_saving_path",
    "_make_saving_path",
    "_fetch_saving_paths",
    "_remove",
    "_rmtree",
]
