import os
import sys
import torch
import cflearn

from cflearn.misc.toolkit import eval_context
from cflearn.misc.toolkit import download_reference


# prepare
folder = os.path.dirname(__file__)
repo_name = "U-2-Net-master"
repo_path = os.path.join(folder, repo_name)
os.system("wget https://github.com/xuebinqin/U-2-Net/archive/refs/heads/master.zip")
os.system(f"unzip {repo_name}.zip -d {folder}")
sys.path.insert(0, repo_path)

from model import U2NET
from model import U2NETP

img_size = 320
in_channels = 3

torch.manual_seed(142857)
inp = torch.randn(1, in_channels, img_size, img_size)

model = U2NET(3, 1)
model_lite = U2NETP(3, 1)
model.load_state_dict(torch.load(download_reference("u2net"), map_location="cpu"))
model_lite.load_state_dict(torch.load(download_reference("u2netp"), map_location="cpu"))

cf_model = cflearn.api.u2net(pretrained=True).model
cf_model_lite = cflearn.api.u2net_lite(pretrained=True).model

with eval_context(model):
    o1s = model(inp)
with eval_context(model_lite):
    o2s = model_lite(inp)
with eval_context(cf_model):
    cfo1s = cf_model(0, {"input": inp})["predictions"]
with eval_context(cf_model_lite):
    cfo2s = cf_model_lite(0, {"input": inp})["predictions"]
for check_idx in range(7):
    o1 = o1s[check_idx]
    o2 = o2s[check_idx]
    cfo1 = torch.sigmoid(cfo1s[check_idx])
    cfo2 = torch.sigmoid(cfo2s[check_idx])
    assert torch.allclose(o1, cfo1)
    assert torch.allclose(o2, cfo2)
