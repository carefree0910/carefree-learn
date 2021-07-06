import os
import numpy as np

from cftool.misc import hash_code

from ...constants import WARNING_PREFIX

try:
    import SharedArray as sa
except:
    sa = None


def _to_address(path: str) -> str:
    return f"shared_{hash_code(path)}"


def _check_sa(address: str) -> bool:
    if sa is None:
        return False
    try:
        sa.attach(address)
        return True
    except FileNotFoundError:
        return False


def _write_shared(address: str, array: np.ndarray) -> None:
    try:
        shared = sa.attach(address)
    except FileNotFoundError:
        shared = sa.create(address, array.shape, array.dtype)
    shared[:] = array


class SharedArrayWrapper:
    def __init__(self, root: str, path: str, *, to_memory: bool):
        self.path = os.path.join(root, path)
        self.folder, file = os.path.split(self.path)
        os.makedirs(self.folder, exist_ok=True)
        self.flag_path = os.path.join(self.folder, f"flag_of_{file}")
        self.address, self.flag_address = map(_to_address, [self.path, self.flag_path])
        if to_memory and sa is None:
            print(
                f"{WARNING_PREFIX}`to_memory` is set to True but `SharedArray` lib "
                f"is not available, therefore `to_memory` will be set to False"
            )
            to_memory = False
        self.to_memory = to_memory

    @property
    def is_ready(self) -> bool:
        if self.to_memory:
            if not _check_sa(self.address):
                return False
            return sa.attach(self.flag_address).item()
        if not os.path.isfile(self.path):
            return False
        if not os.path.isfile(self.flag_path):
            return False
        return bool(np.load(self.flag_path, mmap_mode="r").item())

    def read(self, *, writable: bool = False) -> np.ndarray:
        if self.to_memory:
            arr = sa.attach(self.address)
            arr.flags.writeable = writable
            return arr
        return np.load(self.path, mmap_mode="r+" if writable else "r")

    def write(self, arr: np.ndarray) -> None:
        self._write(arr, overwrite=True, is_finished=True)

    def prepare(self, arr: np.ndarray) -> None:
        # prepare an empty array at certain path / address
        # this is mainly for multiprocessing
        self._write(arr, overwrite=False, is_finished=False)

    def mark_finished(self) -> None:
        flag = self.read(writable=True)
        flag[0] = True

    def delete(self) -> None:
        if self.is_ready:
            print(f"> removing {self.path} & {self.flag_path}")
            if self.to_memory:
                sa.delete(self.address)
                sa.delete(self.flag_address)
                return None
            os.remove(self.path)
            os.remove(self.flag_path)

    def _give_permission(self) -> None:
        os.system(f"chmod -R 777 {self.folder}")

    def _write(
        self,
        arr: np.ndarray,
        *,
        is_finished: bool,
        overwrite: bool = True,
    ) -> None:
        if self.is_ready and overwrite:
            path = self.address if self.to_memory else self.path
            print(
                f"> there's already an array at '{path}', "
                "it will be overwritten"
            )
            self.delete()
        if self.to_memory:
            _write_shared(self.address, arr)
            _write_shared(self.flag_address, np.array([is_finished]))
            return None
        np.save(self.path, arr)
        np.save(self.flag_path, np.array([is_finished]))
        self._give_permission()


__all__ = [
    "SharedArrayWrapper",
]
