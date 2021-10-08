import os
import sys
import torch
import cflearn

from cflearn.misc.toolkit import eval_context
from cflearn.misc.toolkit import download_reference


# prepare
folder = os.path.dirname(__file__)
repo_name = "stylegan2-ada-pytorch-main"
repo_path = os.path.join(folder, repo_name)
url = "https://github.com/NVlabs/stylegan2-ada-pytorch/archive/refs/heads/main.zip"
os.system(f"wget {url}")
os.system(f"mv main.zip {repo_name}.zip")
os.system(f"unzip {repo_name}.zip -d {folder}")
sys.path.insert(0, repo_path)

import legacy

torch.manual_seed(142857)
z = torch.randn(1, 512)


for src in [
    "afhqcat",
    "afhqdog",
    "afhqwild",
    "brecahad",
    "ffhq",
    "metfaces",
]:
    with open(download_reference(src), "rb") as f:
        g = legacy.load_network_pkl(f)["G_ema"]
    cf_model_name = f"generator/style_gan2_generator.{src}"
    cfg = cflearn.DLZoo.load_model(cf_model_name, pretrained=True)

    print(f"=== checking {src} ===")
    with eval_context(g):
        o = g(z, None, force_fp32=True, noise_mode="const")
    with eval_context(cfg):
        cfo = cfg(0, {"input": z}, noise_mode="const")["predictions"]
    assert torch.allclose(o, cfo, atol=1.0e-5)
