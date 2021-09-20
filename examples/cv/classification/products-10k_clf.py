# type: ignore

import os
import cflearn

from typing import List
from cflearn.misc.toolkit import check_is_ci
from cflearn.misc.toolkit import download_dataset


is_ci = check_is_ci()

data_folder = "../data" if is_ci else "data"
dataset = f"products-10k{'_clf_tiny' if is_ci else ''}"
src_folder = os.path.join(data_folder, dataset)
tgt_folder = os.path.join(data_folder, "products-10k_data")


def prepare() -> None:
    def label_fn(hierarchy: List[str]) -> int:
        return int(hierarchy[2] == "pos")

    if is_ci and not os.path.isdir(src_folder):
        download_dataset(dataset, root=data_folder)
    cflearn.cv.prepare_image_folder(
        src_folder,
        tgt_folder,
        to_index=False,
        label_fn=label_fn,
        make_labels_in_parallel=False,
        num_jobs=0 if is_ci else 8,
    )


if __name__ == "__main__":
    prepare()
    data = cflearn.cv.ImageFolderData(
        tgt_folder,
        batch_size=16,
        num_workers=4,
        transform=["resize", "to_tensor"],
    )

    m = cflearn.cv.CarefreePipeline(
        "clf",
        {
            # "img_size": 28,
            "in_channels": 4,
            "num_classes": 2,
            "latent_dim": 512,
            "encoder1d": "backbone",
            "encoder1d_config": {"name": "resnet18"},
        },
        loss_name="cross_entropy",
        metric_names=["acc"] if is_ci else ["acc", "auc"],
        callback_names=["clf", "mlflow"],
        callback_configs={"mlflow": {"experiment_name": "products-10k_clf"}},
        fixed_steps=1 if is_ci else None,
    )
    m.fit(data, cuda=None if is_ci else 7)
