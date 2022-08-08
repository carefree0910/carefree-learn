from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.types import np_dict_type

from ....data import Transforms
from ....data import InferenceImageFolderData
from ....pipeline import DLPipeline


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
    **predict_kwargs: Any,
) -> FolderInferenceResults:
    idata = InferenceImageFolderData(
        folder,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform,
        transform_config=transform_config,
    )
    outputs = m.predict(idata, use_tqdm=use_tqdm, **predict_kwargs)
    return FolderInferenceResults(outputs, idata.dataset.img_paths)


__all__ = [
    "predict_folder",
]
