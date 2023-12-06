# type: ignore

import cflearn
import requests

import numpy as np

from PIL import Image
from transformers import CLIPModel
from transformers import CLIPProcessor
from cflearn.toolkit import check_is_ci
from cflearn.toolkit import eval_context


is_ci = check_is_ci()
version = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(version)
processor = CLIPProcessor.from_pretrained(version)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a photo of a cat.", "a photo of a dog"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with eval_context(model):
    outputs = model(**inputs)

text_embeds = outputs.text_embeds.numpy()
image_embeds = outputs.image_embeds.numpy()

m = cflearn.zoo.clip()
device = "cpu" if is_ci else "cuda"
cf_clip = cflearn.multimodal.CLIPExtractor(m, device)
cf_text_embeds = cf_clip.get_texts_latent(texts)
cf_image_embeds = cf_clip.get_image_latent(image)

text_diff = np.abs(text_embeds - cf_text_embeds)
image_diff = np.abs(image_embeds - cf_image_embeds)
assert text_diff.max().item() <= 1.0e-6
assert image_diff.max().item() <= 1.0e-6
