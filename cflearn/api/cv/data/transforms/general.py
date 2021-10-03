from .....data import Transforms


class NoBatchTransforms(Transforms):
    @property
    def need_batch_process(self) -> bool:
        return False


__all__ = [
    "NoBatchTransforms",
]
