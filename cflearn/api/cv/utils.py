from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.types import np_dict_type

from .pipeline import CVPipeline
from ...data import Transforms
from ...data import InferenceImagePathsData
from ...data import InferenceImageFolderData


class InferenceResults(NamedTuple):
    outputs: np_dict_type
    img_paths: List[str]


def predict_paths(
    m: CVPipeline,
    paths: List[str],
    *,
    batch_size: int,
    num_workers: int = 0,
    transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
    transform_config: Optional[Dict[str, Any]] = None,
    use_tqdm: bool = True,
    **predict_kwargs: Any,
) -> InferenceResults:
    idata = InferenceImagePathsData(
        paths,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=m.test_transform or transform,
        transform_config=transform_config,
    )
    outputs = m.predict(idata, use_tqdm=use_tqdm, **predict_kwargs)
    return InferenceResults(outputs, idata.dataset.img_paths)


def predict_folder(
    m: CVPipeline,
    folder: str,
    *,
    batch_size: int,
    num_workers: int = 0,
    transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
    transform_config: Optional[Dict[str, Any]] = None,
    use_tqdm: bool = True,
    **predict_kwargs: Any,
) -> InferenceResults:
    idata = InferenceImageFolderData(
        folder,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=m.test_transform or transform,
        transform_config=transform_config,
    )
    outputs = m.predict(idata, use_tqdm=use_tqdm, **predict_kwargs)
    return InferenceResults(outputs, idata.dataset.img_paths)


__all__ = [
    "predict_paths",
    "predict_folder",
]
