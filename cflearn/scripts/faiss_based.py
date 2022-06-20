import os
import json
import shutil

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Callable
from typing import Optional

from ..types import tensor_dict_type
from ..trainer import DeviceInfo
from ..constants import WARNING_PREFIX
from ..api.cv.pipeline import SimplePipeline
from ..api.cv.models.interface import IImageExtractor

try:
    import faiss
except ImportError:
    faiss = None


def _check() -> None:
    if faiss is None:
        msg = f"{WARNING_PREFIX}`faiss` is needed for faiss based scripts"
        raise ValueError(msg)


def run_faiss(
    x: np.ndarray,
    index_path: str,
    *,
    dimension: int,
    factory: str = "IVF128,Flat",
) -> Any:
    _check()
    index = faiss.index_factory(dimension, factory)
    print(">> training index")
    index.train(x)
    print(">> adding data to index")
    index.add(x)
    print(">> saving index")
    faiss.write_index(index, index_path)
    print("> done")


def image_retrieval(
    m: SimplePipeline,
    packed: str,
    extractor: IImageExtractor,
    *,
    tag: str,
    task: str,
    data_folder: str,
    input_sample: tensor_dict_type,
    index_dimension: int,
    path_converter: Callable[[str], str],
    batch_size: int = 128,
    num_workers: int = 32,
    index_factory: str = "IVF128,Flat",
    forward_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    output_names: Optional[List[str]] = None,
    cuda: int = 7,
) -> None:
    _check()
    version_folder = os.path.join(".versions", task, tag)
    features_folder = os.path.join(version_folder, "features")
    dist_folder = os.path.join(version_folder, "dist")
    model_file = f"{tag}.zip"
    onnx_file = f"{tag}.onnx"
    features_file = "features.npy"
    files_file = "files.json"

    if os.path.isdir(version_folder):
        print(f"> Warning : '{version_folder}' already exists, it will be removed")
        shutil.rmtree(version_folder)
    m.device_info = DeviceInfo(str(cuda), None)
    m.model.to(f"cuda:{cuda}")
    m.to_onnx(
        version_folder,
        onnx_file=onnx_file,
        forward_fn=forward_fn,
        input_sample=input_sample,
        output_names=output_names,
    )

    os.makedirs(features_folder, exist_ok=True)
    kw = dict(batch_size=batch_size, num_workers=num_workers)
    files = []
    features = []
    for split in ["train", "valid"]:
        split_folder = os.path.join(data_folder, split)
        with open(os.path.join(split_folder, "path_mapping.json"), "r") as f:
            mapping = json.load(f)
        rs = extractor.get_folder_latent(split_folder, **kw)  # type: ignore
        files.extend([mapping[file] for file in rs[1]])
        features.append(rs[0])

    x = np.vstack(features)
    np.save(os.path.join(features_folder, features_file), x)
    files_path = os.path.join(features_folder, files_file)
    with open(files_path, "w") as f:
        json.dump([path_converter(file) for file in files], f, ensure_ascii=False)

    index = faiss.index_factory(index_dimension, index_factory)
    print(">> training index")
    index.train(x)
    print(">> adding data to index")
    index.add(x)
    print(">> saving index")
    index_file = f"{task}.{tag}.index"
    index_path = os.path.join(features_folder, index_file)
    faiss.write_index(index, index_path)
    print("> done")

    os.makedirs(dist_folder)
    print(">> copying model")
    shutil.copyfile(f"{packed}.zip", os.path.join(dist_folder, model_file))
    print(">> copying onnx")
    shutil.copyfile(
        os.path.join(version_folder, onnx_file),
        os.path.join(dist_folder, onnx_file),
    )
    print(">> copying files")
    shutil.copyfile(files_path, os.path.join(dist_folder, files_file))
    print(">> copying index")
    shutil.copyfile(index_path, os.path.join(dist_folder, index_file))
    print("> done")


__all__ = [
    "run_faiss",
    "image_retrieval",
]
