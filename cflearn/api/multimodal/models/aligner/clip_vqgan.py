from typing import Any
from typing import Union
from typing import Tuple
from typing import Optional

from ....dl.data import DummyData
from ....zoo.core import DLZoo


class CLIPWithVQGANTrainer:
    def __init__(
        self,
        text: str,
        *,
        resolution: Tuple[int, int] = (400, 224),
        condition_path: Optional[str] = None,
        **kwargs: Any,
    ):
        kwargs["model_config"] = {
            "text": text,
            "resolution": resolution,
            "condition_path": condition_path,
        }
        self.m = DLZoo.load_pipeline("multimodal/clip_vqgan_aligner", **kwargs)

    def run(self, *, cuda: Optional[Union[int, str]]) -> None:
        self.m.fit(DummyData(), cuda=cuda)  # type: ignore


__all__ = [
    "CLIPWithVQGANTrainer",
]
