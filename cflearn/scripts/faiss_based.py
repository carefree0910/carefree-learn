import os
import json
import torch
import shutil
import tempfile

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.types import tensor_dict_type

from ..schema import DeviceInfo
from ..constants import INPUT_KEY
from ..constants import PREDICTIONS_KEY
from ..api.api import clip
from ..api.api import load
from ..api.schema import IImageExtractor
from ..api.multimodal import CLIPExtractor
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
    # 1) if returns `str`, it will be saved as-is
    # 2) if returns `Dict`, it will be saved with `json.dumps`
    info_fn: Callable[[str], Union[str, Dict[str, Any]]],
    batch_size: int = 128,
    num_workers: int = 32,
    is_raw_data_folder: bool = False,
    index_factory: str = "IVF128,Flat",
    use_cosine_similarity: bool = False,
    forward_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    output_names: Optional[List[str]] = None,
    cuda: Optional[int] = None,
) -> str:
    _check()
    version_folder = os.path.join(".versions", task, tag)
    features_folder = os.path.join(version_folder, "features")
    dist_folder = os.path.join(version_folder, "dist")
    onnx_file = f"{tag}.onnx"
    features_file = "features.npy"
    info_file = "info.json"

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
        img_paths = rs.img_paths
    else:
        xs = []
        img_paths = []
        for split in ["train", "valid"]:
            split_folder = os.path.join(data_folder, split)
            with open(os.path.join(split_folder, "path_mapping.json"), "r") as f:
                mapping = json.load(f)
            rs = extractor.get_folder_latent(split_folder, **kw)  # type: ignore
            xs.append(rs.latent)
            img_paths.extend([mapping[path] for path in rs.img_paths])
        x = np.vstack(xs)
    np.save(os.path.join(features_folder, features_file), x)
    info_path = os.path.join(features_folder, info_file)
    with open(info_path, "w") as f:
        info_list = list(map(json.dumps, map(info_fn, img_paths)))
        json.dump(info_list, f, ensure_ascii=False)

    index_file = f"{task}.{tag}.index"
    index_path = os.path.join(features_folder, index_file)
    build_faiss(
        x,
        index_path,
        dimension=index_dimension,
        factory=index_factory,
        use_cosine_similarity=use_cosine_similarity,
    )

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
    print(">> copying info")
    shutil.copyfile(info_path, os.path.join(dist_folder, info_file))
    print(">> copying index")
    shutil.copyfile(index_path, os.path.join(dist_folder, index_file))
    print("> done")

    return dist_folder


class IRResponse(NamedTuple):
    scores: List[float]
    info_list: List[Any]


def clip_image_retrieval(
    *,
    tag: str,
    task: str,
    data_folder: str,
    info_fn: Callable[[str], str],
    index_factory: str = "IVF128,Flat",
    cuda: Optional[int] = None,
) -> str:
    _check()
    m = clip()
    with tempfile.TemporaryDirectory() as tmp_dir:
        packed = os.path.join(tmp_dir, "packed")
        m.save(packed, compress=False)
        model = m.model
        img_size = model.img_size
        return image_retrieval(
            m,
            packed,
            CLIPExtractor(m),
            tag=tag,
            task=task,
            data_folder=data_folder,
            input_sample={INPUT_KEY: torch.zeros(1, 3, img_size, img_size)},
            index_dimension=512,
            info_fn=info_fn,
            is_raw_data_folder=True,
            index_factory=index_factory,
            use_cosine_similarity=True,
            forward_fn=lambda b: model.encode_image(b[INPUT_KEY]),
            output_names=[PREDICTIONS_KEY],
            cuda=cuda,
        )


def test_clip_image_retrieval(
    dist_folder: str,
    *,
    test_data_folder: str,
    top_k: int = 16,
    n_probe: Optional[int] = None,
    batch_size: int = 16,
    cuda: Optional[int] = None,
) -> Dict[str, IRResponse]:
    # collect
    index_path = None
    for file in os.listdir(dist_folder):
        path = os.path.join(dist_folder, file)
        if file.endswith(".index"):
            index_path = path
    if index_path is None:
        raise ValueError(f"index is not found under '{dist_folder}'")
    info_path = os.path.join(dist_folder, "info.json")
    if not os.path.isfile(info_path):
        raise ValueError(f"'{info_path}' is not found under '{dist_folder}'")
    with open(info_path, "r") as f:
        pool = list(map(json.loads, json.load(f)))
    # inference
    m = load(os.path.join(dist_folder, "packed"), cuda=cuda, compress=False)
    clip_extractor = CLIPExtractor(m)
    clip_rs = clip_extractor.get_folder_latent(test_data_folder, batch_size=batch_size)
    latent = clip_rs.latent
    img_paths = clip_rs.img_paths
    # predict
    final = {}
    api = FaissAPI(index_path)
    faiss_rs = api.predict(latent, top_k=top_k, n_probe=n_probe)
    for i, (i_indices, i_metrics) in enumerate(zip(faiss_rs.indices, faiss_rs.metrics)):
        final[img_paths[i]] = IRResponse(i_metrics, [pool[idx] for idx in i_indices])
    return final
