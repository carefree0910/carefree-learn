import cflearn
import requests

import numpy as np

from PIL import Image
from transformers import CLIPModel
from transformers import CLIPProcessor
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from cflearn.misc.toolkit import eval_context


# text
tag = "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese"
text_tokenizer = BertTokenizer.from_pretrained(tag)
text_encoder = BertForSequenceClassification.from_pretrained(tag)

zh = ["一只小猫", "一只小狗"]
with eval_context(text_encoder):
    text = text_tokenizer(zh, return_tensors="pt", padding=True)["input_ids"]
    bert_logits = text_encoder(text).logits
    text_embeds = bert_logits / bert_logits.norm(dim=1, keepdim=True)
    text_embeds = text_embeds.numpy()

# image
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with eval_context(model):
    outputs = model(**inputs)

image_embeds = outputs.image_embeds.numpy()

# cflearn
m = cflearn.api.chinese_clip()
cf_clip = cflearn.multimodal.CLIPExtractor(m)
cf_text_embeds = cf_clip.get_texts_latent(zh)
cf_image_embeds = cf_clip.get_image_latent(image)

text_diff = np.abs(text_embeds - cf_text_embeds)
image_diff = np.abs(image_embeds - cf_image_embeds)
assert text_diff.max().item() <= 1.0e-6
assert image_diff.max().item() <= 1.0e-6
