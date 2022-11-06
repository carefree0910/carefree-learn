import os
import cflearn
import requests

import numpy as np

from PIL import Image
from transformers import CLIPModel
from transformers import CLIPProcessor
from cflearn.misc.toolkit import eval_context


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a photo of a cat.", "a photo of a dog"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with eval_context(model):
    outputs = model(**inputs)

text_embeds = outputs.text_embeds.numpy()
image_embeds = outputs.image_embeds.numpy()

m = cflearn.api.clip()
cf_clip = cflearn.multimodal.CLIPExtractor(m)
os.makedirs(".tmp", exist_ok=True)
image.save(f".tmp/test.png")
cf_text_embeds = cf_clip.get_texts_latent(texts)
cf_image_embeds = cf_clip.get_folder_latent(".tmp", batch_size=1).latent

text_diff = np.abs(text_embeds - cf_text_embeds)
image_diff = np.abs(image_embeds - cf_image_embeds)
assert text_diff.max().item() <= 1.0e-6
assert image_diff.max().item() <= 1.0e-6
