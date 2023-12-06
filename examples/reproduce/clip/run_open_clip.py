import cflearn
import requests
import open_clip

import numpy as np

from PIL import Image
from cflearn.toolkit import eval_context

model_name = "ViT-H-14"
pretrained = "laion2b_s32b_b79k"

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name,
    pretrained=pretrained,
)
model.eval()
tokenizer = open_clip.get_tokenizer(model_name)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a photo of a cat.", "a photo of a dog"]

image_tensor = preprocess(image).unsqueeze(0)
text_tensor = tokenizer(texts)

with eval_context(model):
    image_embeds = model.encode_image(image_tensor)
    text_embeds = model.encode_text(text_tensor)
    image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
    text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
image_embeds = image_embeds.numpy()
text_embeds = text_embeds.numpy()

m = cflearn.zoo.open_clip_ViT_H_14()
cf_clip = cflearn.multimodal.CLIPExtractor(m, "cuda")
cf_text_embeds = cf_clip.get_texts_latent(texts)
cf_image_embeds = cf_clip.get_image_latent(image)

text_diff = np.abs(text_embeds - cf_text_embeds)
image_diff = np.abs(image_embeds - cf_image_embeds)
assert text_diff.max().item() <= 1.0e-6
assert image_diff.max().item() <= 1.0e-6
