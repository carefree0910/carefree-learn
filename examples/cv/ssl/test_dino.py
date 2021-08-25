import os
import json
import torch

import numpy as np

from PIL import Image
from cftool.dist import Parallel
from pykeops.torch import LazyTensor
from cflearn.misc.toolkit import to_torch
from cflearn.misc.toolkit import save_images

similar = True
src = ".features"
tgt = ".visualizations"

query = "valid"
anchor = "train"

query_features = torch.load(os.path.join(src, f"{query}.pt"))
anchor_features = torch.load(os.path.join(src, f"{anchor}.pt"))
with open(os.path.join(src, f"{query}.json"), "r") as f:
    query_paths = json.load(f)
with open(os.path.join(src, f"{anchor}.json"), "r") as f:
    anchor_paths = json.load(f)

query_lazy = LazyTensor(query_features[:, None, :])
anchor_lazy = LazyTensor(anchor_features[None, ...])
query_anchor_pair = ((query_lazy - anchor_lazy) ** 2).sum(2)
if not similar:
    query_anchor_pair = -query_anchor_pair

tgt = os.path.join(tgt, "similar" if similar else "difference")
os.makedirs(tgt, exist_ok=True)
candidates = query_anchor_pair.argKmin(8 + int(similar), dim=1)


def to_torch_img(path: str) -> torch.Tensor:
    arr = np.array(Image.open(path).resize((320, 320)).convert("RGB"))
    return to_torch(arr.transpose([2, 0, 1]))


def task(iq: int, query_path: str) -> None:
    images = [to_torch_img(query_path)]
    idx_candidates = candidates[iq][int(similar) :].tolist()
    for j_idx in idx_candidates:
        images.append(to_torch_img(anchor_paths[j_idx]))
    image = torch.stack(images, dim=0).to(torch.float32) / 255.0
    save_images(image, os.path.join(tgt, f"{iq}.png"))


Parallel(16).grouped(task, list(range(len(query_paths))), query_paths)
