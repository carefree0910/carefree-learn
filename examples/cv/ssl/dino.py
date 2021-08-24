import os
import cflearn
import argparse

from PIL import Image
from typing import List
from cflearn.misc.toolkit import download_dataset


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
        num_workers=0 if is_ci else 10,
        transform="ssl",
        transform_config={"img_size": img_size},
    )

    m = cflearn.cv.CarefreePipeline(
        "dino",
        {
            "out_dim": 512,
            "norm_last_layer": False,
            "teacher_temp_epochs": num_epoch,
            "encoder_configs": {"img_size": img_size, "in_channels": 3},
        },
        amp=True,
        fixed_epoch=num_epoch,
        callback_names=["mlflow"],
        callback_configs={"mlflow": {"experiment_name": "poster"}},
        fixed_steps=1 if is_ci else None,
    )
    m.fit(data, cuda=None if is_ci else 0)
