import os
import torch
import cflearn

import numpy as np

from PIL import Image
from cflearn.api.cv.diffusion import ControlNetHints


file_folder = os.path.dirname(__file__)
api = cflearn.cv.ControlledDiffusionAPI.from_sd("cuda:0", use_half=True)
# prepare ControlNet weights/annotators
api.prepare_defaults()
api.prepare_annotators()
# for ControlledDiffusionAPI, we need to disable `ControlNet` before doing normal generations
api.disable_control()
api.txt2img("A lovely little cat.", "out.png", seed=234)
# and we can enable `ControlNet` if we need it
# > Notice that disable/enable `ControlNet` is very fast so you can use it calmly!
api.enable_control()
# switch to `canny` hint type and get canny hint
cat_path = f"{file_folder}/assets/cat.png"
api.switch(ControlNetHints.CANNY)
hint = api.get_hint_of(
    ControlNetHints.CANNY,
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
    hint={ControlNetHints.CANNY: hint_tensor},
    seed=234,
)
api.img2img(
    cat_path,
    "controlled_img2img.png",
    cond=["A lovely little cat."],
    hint={ControlNetHints.CANNY: hint_tensor},
    seed=234,
)
