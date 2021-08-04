# type: ignore

import os
import cv2
import cflearn

import numpy as np

from typing import List
from cflearn.misc.toolkit import min_max_normalize


src_folder = "data/human_matting"
tgt_folder = "data/human_matting_data"
label_folder = "data/human_matting_labels"


def prepare() -> None:
    def label_fn(hierarchy: List[str]) -> str:
        matting_path = os.path.abspath(
            os.path.join(
                hierarchy[0],
                hierarchy[1],
                "matting",
                hierarchy[3],
                hierarchy[4].replace("clip", "matting"),
                hierarchy[5].replace(".jpg", ".png"),
            )
        )
        alpha = cv2.imread(matting_path, cv2.IMREAD_UNCHANGED)[..., -1:]
        alpha = min_max_normalize(alpha.astype(np.float32), global_norm=True)
        file = hierarchy[-1]
        file_id = os.path.splitext(file)[0]
        os.makedirs(label_folder, exist_ok=True)
        label_path = os.path.join(label_folder, f"{file_id}.npy")
        label_path = os.path.abspath(label_path)
        if os.path.isfile(label_path):
            return label_path
        np.save(label_path, alpha)
        return label_path

    def filter_fn(hierarchy: List[str]) -> bool:
        return hierarchy[2] == "clip_img"

    cflearn.cv.prepare_image_folder(
        src_folder,
        tgt_folder,
        to_index=False,
        label_fn=label_fn,
        filter_fn=filter_fn,
        make_labels_in_parallel=True,
    )


if __name__ == "__main__":
    prepare()
    train_loader, valid_loader = cflearn.cv.get_image_folder_loaders(
        tgt_folder,
        batch_size=8,
        num_workers=2,
        transform="for_salient_object_detection",
    )
    cflearn.MultiStageLoss.register_(["bce", "iou", "sigmoid_mae"])
    m = cflearn.cv.CarefreePipeline(
        "u2net",
        {
            "in_channels": 3,
            "out_channels": 1,
        },
        loss_name="multi_stage_bce_iou_sigmoid_mae",
        loss_metrics_weights={"bce0": 0.2, "iou0": 0.4, "sigmoid_mae0": 0.4},
        callback_names=["u2net", "mlflow"],
        callback_configs={"mlflow": {"experiment_name": "large_hm"}},
    )
    m.fit(train_loader, valid_loader, cuda="0")
