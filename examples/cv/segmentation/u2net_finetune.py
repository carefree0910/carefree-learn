# type: ignore

import os
import cv2
import torch
import cflearn

import numpy as np
import torch.nn.functional as F

from typing import Any
from typing import List
from typing import Optional
from cflearn.types import losses_type
from cflearn.types import tensor_dict_type
from cflearn.protocol import LossProtocol
from cflearn.protocol import TrainerState
from cflearn.constants import LABEL_KEY
from cflearn.constants import PREDICTIONS_KEY
from cflearn.misc.toolkit import min_max_normalize


src_folder = "raw"
src_rgba_folder = "rgba"
tgt_folder = "u2net_finetune"
label_folder = "u2net_finetune_labels"


def prepare() -> None:
    def label_fn(hierarchy: List[str]) -> str:
        file_id = os.path.splitext(hierarchy[-1])[0]
        os.makedirs(label_folder, exist_ok=True)
        label_path = os.path.abspath(os.path.join(label_folder, f"{file_id}.npy"))
        if os.path.isfile(label_path):
            return label_path
        hierarchy[0] = src_rgba_folder
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


@LossProtocol.register("sigmoid_mae")
class SigmoidMAE(LossProtocol):
    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        losses = F.l1_loss(torch.sigmoid(predictions), labels, reduction="none")
        return losses.mean((1, 2, 3))


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
            # "lite": True,
        },
        loss_name="multi_stage_bce_iou_sigmoid_mae",
        loss_metrics_weights={"bce0": 0.2, "iou0": 0.4, "sigmoid_mae0": 0.4},
        callback_names=["u2net", "mlflow"],
        callback_configs={"mlflow": {"experiment_name": "large_pretrain"}},
        # clip_norm=1.0,
        # lr=4.0e-3,
        # scheduler_name="none",
        # optimizer_config={"weight_decay": 1.0e-4},
        finetune_config={
            "pretrained_ckpt": "pretrained/model.pt",
            # "pretrained_ckpt": "pretrained/model_lite.pt",
            # "freeze_except": r"(.*\.side_blocks\..*|.*\.out\..*)",
        },
    )
    m.fit(train_loader, valid_loader, cuda="3")
    # m.ddp(train_loader, valid_loader, cuda_list=[0, 2, 3, 4])
