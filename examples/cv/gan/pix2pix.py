import os
import random
import cflearn

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from cftool.misc import hash_code
from cftool.array import to_device
from cftool.array import save_images
from cftool.types import tensor_dict_type
from cflearn.constants import INPUT_KEY
from cflearn.constants import LABEL_KEY
from cflearn.misc.toolkit import eval_context
from torchvision.transforms import InterpolationMode


task = "edges2shoes"

src_folder = f"./datasets/{task}/train"
tgt_folder = f"./datasets/cf_{task}/src"
label_folder = f"./datasets/cf_{task}/tgt"
os.makedirs(label_folder, exist_ok=True)


def split(path: str) -> Tuple[Image.Image, Image.Image]:
    concat = Image.open(path).convert("RGB")
    w, h = concat.size
    w2 = int(w / 2)
    src = concat.crop((0, 0, w2, h))
    tgt = concat.crop((w2, 0, w, h))
    return src, tgt


class Pix2PixPreparation(cflearn.DefaultPreparation):
    def copy(self, src_path: str, tgt_path: str) -> None:
        split(src_path)[0].save(tgt_path)

    def get_label(self, hierarchy: List[str]) -> str:
        code = hash_code("".join(hierarchy))
        label_path = os.path.abspath(os.path.join(label_folder, f"{code}.npy"))
        np.save(label_path, np.array(split(os.path.join(*hierarchy))[1]))
        return label_path


@cflearn.register_transform("pix2pix")
class Pix2PixITransform(cflearn.ITransform):
    def __init__(self, *, resize_size: int, crop_size: int):
        self.resize_size = resize_size
        self.crop_size = crop_size

    def forward(self, sample: Dict[str, Any]) -> tensor_dict_type:
        src = sample[INPUT_KEY]
        tgt = Image.fromarray(sample[LABEL_KEY])
        x = random.randint(0, np.maximum(0, self.resize_size - self.crop_size))
        y = random.randint(0, np.maximum(0, self.resize_size - self.crop_size))
        flip = random.random() > 0.5
        transform_list = [
            transforms.Resize(
                [self.resize_size, self.resize_size], InterpolationMode.BICUBIC
            ),
            transforms.Lambda(
                lambda img: img.crop((x, y, x + self.crop_size, y + self.crop_size))
            ),
        ]
        if flip:
            transform_list.append(
                transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT))
            )
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transform = transforms.Compose(transform_list)
        src, tgt = map(transform, [src, tgt])
        return {INPUT_KEY: src, LABEL_KEY: tgt}


@cflearn.register_transform("pix2pix_test")
class Pix2PixTestITransform(cflearn.ITransform):
    def __init__(self, *, size: int):
        self.size = size

    def forward(self, sample: Dict[str, Any]) -> tensor_dict_type:
        src = sample[INPUT_KEY]
        tgt = Image.fromarray(sample[LABEL_KEY])
        transform = transforms.Compose(
            [
                transforms.Resize([self.size, self.size], InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        src, tgt = map(transform, [src, tgt])
        return {INPUT_KEY: src, LABEL_KEY: tgt}


@cflearn.ImageCallback.register("pix2pix")
class StyleTransferCallback(cflearn.ImageCallback):
    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        src = batch[INPUT_KEY]
        tgt = batch[LABEL_KEY]
        image_folder = self._prepare_folder(trainer)
        # inputs
        save_images(src, os.path.join(image_folder, "src.png"))
        save_images(tgt, os.path.join(image_folder, "tgt.png"))
        # stylize
        model = trainer.model.core.generator
        with eval_context(model):
            translated = model(src)
        save_images(translated, os.path.join(image_folder, "translated.png"))


if __name__ == "__main__":
    prep = cflearn.prepare_image_folder_data(
        src_folder,
        tgt_folder,
        to_index=False,
        preparation=Pix2PixPreparation(),
        batch_size=1,
        num_workers=1,
        transform="pix2pix",
        transform_config=dict(resize_size=286, crop_size=256),
        test_transform="pix2pix_test",
        test_transform_config=dict(size=256),
        make_labels_in_parallel=True,
        num_jobs=32,
        valid_split=0.0,
    )

    m = cflearn.DLZoo.load_pipeline("gan/pix2pix")
    m.fit(prep.data, cuda=0)
