import os

import json
import torch
import cflearn

from PIL import Image


base = cflearn.cv.CarefreePipeline
packed = base.pack(".versions/512_2000")
dino = base.load(packed).model.cuda()
dino_api = cflearn.cv.DINOPredictor(dino)

data_folder = "data/poster_data"
features_folder = ".features"
Image.MAX_IMAGE_PIXELS = None

train_data_folder = os.path.join(data_folder, "train")
valid_data_folder = os.path.join(data_folder, "valid")

os.makedirs(features_folder, exist_ok=True)
kw = dict(batch_size=4, num_workers=2)
rs = dino_api.get_folder_latent(train_data_folder, **kw)  # type: ignore
torch.save(rs[0], os.path.join(features_folder, "train.pt"))
with open(os.path.join(features_folder, "train.json"), "w") as f:
    json.dump(rs[1], f)
rs = dino_api.get_folder_latent(valid_data_folder, **kw)  # type: ignore
torch.save(rs[0], os.path.join(features_folder, "valid.pt"))
with open(os.path.join(features_folder, "valid.json"), "w") as f:
    json.dump(rs[1], f)
