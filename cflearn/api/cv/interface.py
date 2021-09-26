from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple

from ...data import Transforms
from ...data import InferenceImageFolderData
from ...types import np_dict_type
from ...pipeline import DLPipeline


class FolderInferenceResults(NamedTuple):
    outputs: np_dict_type
    img_paths: List[str]


def predict_folder(
    m: DLPipeline,
    folder: str,
    *,
    batch_size: int,
    num_workers: int = 0,
    transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
    transform_config: Optional[Dict[str, Any]] = None,
    use_tqdm: bool = True,
) -> FolderInferenceResults:
    data = InferenceImageFolderData(
        folder,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform,
        transform_config=transform_config,
    )
    outputs = m.predict(data, use_tqdm=use_tqdm)
    return FolderInferenceResults(outputs, data.dataset.img_paths)


__all__ = [
    "predict_folder",
]
