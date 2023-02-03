import os
import torch
import cflearn
import platform

import numpy as np

from PIL import Image
from typing import Any
from typing import List
from typing import Optional
from cflearn.misc.toolkit import download_model


IS_LINUX = platform.system() == "Linux"


num_samples = 100
data_folder = "static/data"
pretrained_folder = "static/pretrained"
os.makedirs(pretrained_folder, exist_ok=True)


# data
class AuxPreparation(cflearn.DefaultPreparation):
    @property
    def extra_labels(self) -> Optional[List[str]]:
        return ["aux"]

    def get_label(self, hierarchy: List[str]) -> Any:
        idx = os.path.splitext(hierarchy[-1])[0]
        return np.load(os.path.join(labels_folder, f"{idx}.npy")).item()

    def get_extra_label(self, label_name: str, hierarchy: List[str]) -> Any:
        if label_name == "aux":
            idx = os.path.splitext(hierarchy[-1])[0]
            return np.load(os.path.join(aux_labels_folder, f"{idx}.npy")).item()
        raise ValueError(f"label_name '{label_name}' is not recognized")


x = torch.rand(num_samples, 3, 224, 224)
y1 = torch.randint(2, [num_samples, 1]).numpy()
y2 = torch.randint(5, [num_samples, 1]).numpy()
input_folder = os.path.join(data_folder, "input")
labels_folder = os.path.join(data_folder, "labels")
aux_labels_folder = os.path.join(data_folder, "aux_labels")
for folder in (input_folder, labels_folder, aux_labels_folder):
    os.makedirs(folder, exist_ok=True)
for i in range(num_samples):
    uint8 = torch.clamp(x[i] * 255.0, 0.0, 255.0).to(torch.uint8)
    img = Image.fromarray(uint8.numpy().transpose([1, 2, 0]))
    img.save(os.path.join(input_folder, f"{i}.png"))
    np.save(os.path.join(labels_folder, f"{i}.npy"), y1[i])
    np.save(os.path.join(aux_labels_folder, f"{i}.npy"), y2[i])

preparation = AuxPreparation()
kw = dict(
    to_index=False,
    batch_size=16,
    preparation=preparation,
    transform="clf",
    transform_config={"resize_size": 224},
    test_transform="clf_test",
)
if not IS_LINUX:
    kw["num_jobs"] = 0
rs = cflearn.prepare_image_folder_data(data_folder, "static/processed", **kw)  # type: ignore


# load pretrain
d = torch.load(download_model("cct_large_224"))
pop_keys = []
for k, v in d.items():
    if k.startswith("encoder.head.module.projection.linear"):
        pop_keys.append(k)
for k in pop_keys:
    d.pop(k)
pretrained_path = os.path.join(pretrained_folder, "cct_large_224.pt")
torch.save(d, pretrained_path)

# train
num_classes = preparation.get_num_classes(rs.tgt_folder)
main_num_classes = num_classes.pop(cflearn.LABEL_KEY)
m = cflearn.api.cct_large_224(
    main_num_classes,
    aux_num_classes=num_classes,
    model_config={
        "encoder1d_pretrained_path": pretrained_path,
        "encoder1d_pretrained_strict": False,
    },
    callback_names=["clf", "mlflow"],
    fixed_steps=1,
)
m.fit(rs.data, cuda=None)
