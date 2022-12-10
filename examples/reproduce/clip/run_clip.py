import os
import sys
import torch
import cflearn

from cflearn.misc.toolkit import eval_context


# prepare
folder = os.path.dirname(__file__)
repo_name = "CLIP-main"
repo_path = os.path.join(folder, repo_name)
url = "https://github.com/openai/CLIP/archive/refs/heads/main.zip"
os.system(f"wget {url}")
os.system(f"mv main.zip {repo_name}.zip")
os.system(f"unzip {repo_name}.zip -d {folder}")
sys.path.insert(0, repo_path)

import clip

device = "cpu"
cf_tokenizer = cflearn.ITokenizer.make("clip", {})

torch.manual_seed(142857)
img = torch.randn(1, 3, 224, 224)
texts = ["a diagram.", "a dog", "a cat"]
text = clip.tokenize(texts).to(torch.long)
text[text == 0] = cf_tokenizer.eos_token_id
assert torch.allclose(text, torch.from_numpy(cf_tokenizer.tokenize(texts)))

for clip_name, cf_clip_name in zip(
    ["ViT-B/32", "ViT-L/14"],
    ["multimodal/clip", "multimodal/clip.large"],
):
    model, preprocess = clip.load(clip_name, device=device)
    cf_clip = cflearn.DLZoo.load_model(cf_clip_name, pretrained=True)
    cf_clip.logit_scale = model.logit_scale
    with eval_context(model):
        o = model(img, text)[0]
    with eval_context(cf_clip):
        cfo = cf_clip(0, {"input": img, "text": text})["predictions"]
    assert torch.allclose(o, cfo, atol=1.0e-5)
