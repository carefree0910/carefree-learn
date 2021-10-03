from .....data import Transforms


class NoBatchBase(Transforms):
    @property
    def need_batch_process(self) -> bool:
        return False


__all__ = [
    "NoBatchBase",
]
