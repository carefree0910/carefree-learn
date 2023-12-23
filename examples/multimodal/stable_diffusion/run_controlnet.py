import os
import torch
import cflearn

import numpy as np

from PIL import Image
from cflearn.api.multimodal import ControlNetHints


hint_type = ControlNetHints.CANNY
file_folder = os.path.dirname(__file__)
api = cflearn.multimodal.ControlledDiffusionAPI.from_sd(device="cuda:0", use_half=True)
# prepare ControlNet annotators
api.prepare_annotators()
# for ControlledDiffusionAPI, we need to disable `ControlNet` before doing normal generations
api.disable_control()
api.txt2img("A lovely little cat.", "out.png", seed=234)
# and we can enable `ControlNet` if we need it
# > Notice that disable/enable `ControlNet` is very fast so you can use it calmly!
api.enable_control()
# switch to `canny` hint type and get canny hint
cat_path = f"{file_folder}/assets/cat.png"
api.switch_control(hint_type)
hint = api.get_hint_of(
    hint_type,
    np.array(Image.open(cat_path)),
    low_threshold=100,
    high_threshold=200,
)
Image.fromarray(hint).save("control_hint.png")
hint_tensor = torch.from_numpy(hint)[None].permute(0, 3, 1, 2).contiguous() / 255.0
if api.use_half:
    hint_tensor = hint_tensor.half()
hint_tensor = hint_tensor.to(api.device)
# apply canny hint to ControlNet, notice that the only difference is to add a `hint` argument
api.txt2img(
    "A lovely little cat.",
    "controlled_txt2img.png",
    hint=[(hint_type, hint_tensor)],
    hint_start=[0.0],
    hint_end=[1.0],
    seed=234,
)
api.img2img(
    cat_path,
    "controlled_img2img.png",
    cond=["A lovely little cat."],
    hint=[(hint_type, hint_tensor)],
    hint_start=[0.0],
    hint_end=[1.0],
    seed=234,
)
