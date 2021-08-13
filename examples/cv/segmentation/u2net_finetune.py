# type: ignore

import os
import cv2
import cflearn

import numpy as np

from typing import List
from cflearn.misc.toolkit import min_max_normalize


src_folder = "data/raw"
src_rgba_folder = "data/rgba"
tgt_folder = "data/products-10k"
label_folder = "data/products-10k_labels"


def prepare() -> None:
    def label_fn(hierarchy: List[str]) -> str:
        file_id = os.path.splitext(hierarchy[-1])[0]
        os.makedirs(label_folder, exist_ok=True)
        label_path = os.path.abspath(os.path.join(label_folder, f"{file_id}.npy"))
        if os.path.isfile(label_path):
            return label_path
        hierarchy[0] = src_rgba_folder
        hierarchy.pop(1)
        hierarchy[1] = f"{file_id}.png"
        rgba_path = os.path.abspath(os.path.join(*hierarchy))
        alpha = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)[..., -1:]
        alpha = min_max_normalize(alpha.astype(np.float32), global_norm=True)
        np.save(label_path, alpha)
        return label_path

    cflearn.cv.prepare_image_folder(
        src_folder,
        tgt_folder,
        to_index=False,
        label_fn=label_fn,
        make_labels_in_parallel=True,
    )


if __name__ == "__main__":
    prepare()
    train_loader, valid_loader = cflearn.cv.get_image_folder_loaders(
        tgt_folder,
        batch_size=16,
        num_workers=4,
        transform="a_bundle_with_mask",
        test_transform="a_bundle_with_mask_test",
    )
    m = cflearn.cv.CarefreePipeline(
        "u2net",
        {
            "in_channels": 3,
            "out_channels": 1,
            "lite": True,
        },
        loss_name="multi_stage:bce,iou",
        loss_metrics_weights={"bce0": 0.2, "iou0": 0.8},
        callback_names=["u2net", "mlflow"],
        callback_configs={"mlflow": {"experiment_name": "lite_pretrain"}},
        finetune_config={
            # "pretrained_ckpt": "pretrained/model.pt",
            "pretrained_ckpt": "pretrained/model_lite.pt",
            # "freeze_except": r"(.*\.side_blocks\..*|.*\.out\..*)",
        },
        scheduler_name="none",
    )
    m.fit(train_loader, valid_loader, cuda="0")
    # m.ddp(train_loader, valid_loader, cuda_list=[0, 2, 3, 4])
