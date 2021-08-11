# type: ignore

import cflearn

from typing import List


src_folder = "data/products-10k"
tgt_folder = "data/products-10k_data"


def prepare() -> None:
    def label_fn(hierarchy: List[str]) -> int:
        return int(hierarchy[2] == "pos")

    cflearn.cv.prepare_image_folder(
        src_folder,
        tgt_folder,
        to_index=False,
        label_fn=label_fn,
        make_labels_in_parallel=False,
    )


if __name__ == "__main__":
    prepare()
    train_loader, valid_loader = cflearn.cv.get_image_folder_loaders(
        tgt_folder,
        batch_size=16,
        num_workers=4,
        transform="resize",
    )

    m = cflearn.cv.CarefreePipeline(
        "clf",
        {
            # "img_size": 28,
            "in_channels": 4,
            "num_classes": 2,
            "latent_dim": 512,
            "encoder1d": "backbone",
            "encoder1d_configs": {"name": "resnet18"},
        },
        loss_name="cross_entropy",
        metric_names=["acc", "auc"],
        callback_names=["clf", "mlflow"],
        callback_configs={"mlflow": {"experiment_name": "products-10k_clf"}},
    )
    m.fit(train_loader, valid_loader, cuda="7")
