import os
import random
import cflearn
import argparse

from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
from typing import List
from typing import Tuple
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from cflearn.misc.toolkit import download_dataset


class GaussianBlur:
    def __init__(
        self,
        p: float = 0.5,
        radius_min: float = 0.1,
        radius_max: float = 2.0,
    ):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image) -> Image:
        if random.random() > self.prob:
            return img
        r = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


class Solarization:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, img: Image) -> Image:
        if random.random() > self.p:
            return img
        return ImageOps.solarize(img)


class Augmentation:
    def __init__(
        self,
        img_size: int,
        local_crops_number: int = 8,
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
    ):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1,
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # global crop 1
        self.global_transform1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size,
                    scale=global_crops_scale,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                GaussianBlur(1.0),
                normalize,
            ]
        )
        # global crop 2
        self.global_transform2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size,
                    scale=global_crops_scale,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                GaussianBlur(0.1),
                Solarization(0.2),
                normalize,
            ]
        )
        # local crop
        self.local_crops_number = local_crops_number
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    int(img_size * 3 / 7),
                    scale=local_crops_scale,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                GaussianBlur(0.5),
                normalize,
            ]
        )

    def __call__(self, image: Image) -> Image:
        image = image.convert("RGB")
        crops = [self.global_transform1(image), self.global_transform2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


class DINOTransform(cflearn.cv.Transforms):
    def __init__(self, img_size: int):
        super().__init__()
        self.fn = Augmentation(img_size)

    @property
    def need_batch_process(self) -> bool:
        return False


def prepare() -> None:
    def label_fn(_: List[str]) -> int:
        return 0

    if is_ci and not os.path.isdir(src_folder):
        download_dataset(dataset, root=data_folder)
    cflearn.cv.prepare_image_folder(
        src_folder,
        tgt_folder,
        to_index=False,
        label_fn=label_fn,
        make_labels_in_parallel=False,
        num_jobs=0,
    )


# CI
parser = argparse.ArgumentParser()
parser.add_argument("--ci", type=int, default=0)
args = parser.parse_args()
is_ci = bool(args.ci)

data_folder = "../data" if is_ci else "data"
dataset = f"poster{'_tiny' if is_ci else ''}"
src_folder = os.path.join(data_folder, dataset)
tgt_folder = os.path.join(data_folder, "poster_data")

img_size = 224
num_epoch = 2000
Image.MAX_IMAGE_PIXELS = None


if __name__ == "__main__":
    prepare()
    data = cflearn.cv.ImageFolderData(
        tgt_folder,
        batch_size=4 if is_ci else 32,
        num_workers=0 if is_ci else 8,
        transform=DINOTransform(img_size),
    )

    m = cflearn.cv.CarefreePipeline(
        "dino",
        {
            "out_dim": 512,
            "norm_last_layer": False,
            "teacher_temp_epochs": num_epoch,
            "encoder_configs": {"img_size": img_size, "in_channels": 3},
        },
        fixed_epoch=num_epoch,
        callback_names=["mlflow"],
        callback_configs={"mlflow": {"experiment_name": "poster"}},
        fixed_steps=1 if is_ci else None,
    )
    m.fit(data, cuda=None if is_ci else 0)
