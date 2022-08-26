import os
import json
import cflearn

import numpy as np

from PIL import Image


base = cflearn.cv.CVPipeline
packed = base.pack(".versions/512_2000")
dino_api = cflearn.cv.DINOExtractor(base.load(packed, cuda=0), 224)  # type: ignore

data_folder = "data/poster_data"
features_folder = ".features"
Image.MAX_IMAGE_PIXELS = None

train_data_folder = os.path.join(data_folder, "train")
valid_data_folder = os.path.join(data_folder, "valid")

os.makedirs(features_folder, exist_ok=True)
kw = dict(batch_size=4, num_workers=2)
rs = dino_api.get_folder_latent(train_data_folder, **kw)  # type: ignore
np.save(os.path.join(features_folder, "train.npy"), rs.latent)
with open(os.path.join(features_folder, "train.json"), "w") as f:
    json.dump(rs.img_paths, f)
rs = dino_api.get_folder_latent(valid_data_folder, **kw)  # type: ignore
np.save(os.path.join(features_folder, "valid.npy"), rs.latent)
with open(os.path.join(features_folder, "valid.json"), "w") as f:
    json.dump(rs.img_paths, f)
