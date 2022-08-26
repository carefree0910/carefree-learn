import os
import json
import shutil

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.types import tensor_dict_type

from ..protocol import DeviceInfo
from ..api.protocol import IImageExtractor
from ..api.cv.pipeline import CVPipeline

try:
    import faiss
except ImportError:
    faiss = None


def _check() -> None:
    if faiss is None:
        raise ValueError("`faiss` is needed for faiss based scripts")


class FaissResponse(NamedTuple):
    indices: List[List[int]]
    metrics: List[List[float]]


class FaissAPI:
    def __init__(self, index_path: str):
        _check()
        self.index = faiss.read_index(index_path)

    def predict(
        self,
        query: np.ndarray,
        *,
        top_k: int,
        n_probe: Optional[int] = None,
    ) -> FaissResponse:
        if n_probe is not None:
            self.index.nprobe = n_probe
        top_k = min(top_k, self.index.ntotal)
        metrics, indices = self.index.search(query, top_k)
        metrics = metrics.tolist()
        indices = indices.tolist()
        for i, (sub_i, sub_m) in enumerate(zip(indices, metrics)):
            metrics[i] = [d for j, d in enumerate(sub_m) if sub_i[j] != -1]
            indices[i] = [j for j in sub_i if j != -1]
        return FaissResponse(indices, metrics)


def build_faiss(
    x: np.ndarray,
    index_path: str,
    *,
    dimension: int,
    factory: str = "IVF128,Flat",
    use_cosine_similarity: bool = False,
) -> None:
    _check()
    metric = faiss.METRIC_INNER_PRODUCT if use_cosine_similarity else faiss.METRIC_L2
    index = faiss.index_factory(dimension, factory, metric)
    print(">> training index")
    index.train(x)
    print(">> adding data to index")
    index.add(x)
    print(">> saving index")
    faiss.write_index(index, index_path)
    print("> done")


def image_retrieval(
    m: CVPipeline,
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
    is_raw_data_folder: bool = False,
    index_factory: str = "IVF128,Flat",
    index_metrics: Optional[Any] = None,
    forward_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    output_names: Optional[List[str]] = None,
    cuda: Optional[int] = None,
) -> None:
    _check()
    version_folder = os.path.join(".versions", task, tag)
    features_folder = os.path.join(version_folder, "features")
    dist_folder = os.path.join(version_folder, "dist")
    onnx_file = f"{tag}.onnx"
    features_file = "features.npy"
    files_file = "files.json"

    if os.path.isdir(version_folder):
        print(f"> Warning : '{version_folder}' already exists, it will be removed")
        shutil.rmtree(version_folder)
    if cuda is not None:
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

    if is_raw_data_folder:
        rs = extractor.get_folder_latent(data_folder, **kw)  # type: ignore
        x = rs.latent
        files = rs.img_paths
    else:
        xs = []
        files = []
        for split in ["train", "valid"]:
            split_folder = os.path.join(data_folder, split)
            with open(os.path.join(split_folder, "path_mapping.json"), "r") as f:
                mapping = json.load(f)
            rs = extractor.get_folder_latent(split_folder, **kw)  # type: ignore
            xs.append(rs.latent)
            files.extend([mapping[file] for file in rs.img_paths])
        x = np.vstack(xs)
    np.save(os.path.join(features_folder, features_file), x)
    files_path = os.path.join(features_folder, files_file)
    with open(files_path, "w") as f:
        json.dump([path_converter(file) for file in files], f, ensure_ascii=False)

    args = [index_dimension, index_factory]
    if index_metrics is not None:
        args.append(index_metrics)
    index = faiss.index_factory(*args)
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
    if os.path.isdir(packed):
        shutil.copytree(packed, os.path.join(dist_folder, "packed"))
    else:
        model_path = os.path.join(dist_folder, f"{tag}.zip")
        shutil.copyfile(f"{packed}.zip", model_path)
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
    "build_faiss",
    "image_retrieval",
    "FaissAPI",
]
