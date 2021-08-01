# type: ignore

import torch
import cflearn

from typing import Any
from typing import Optional
from torch.nn import L1Loss
from cflearn.types import losses_type
from cflearn.types import tensor_dict_type
from cflearn.constants import LOSS_KEY
from cflearn.constants import LABEL_KEY
from cflearn.constants import PREDICTIONS_KEY
from cflearn.protocol import TrainerState
from cflearn.api.cv import AlphaSegmentationCallback
from cflearn.misc.toolkit import iou

from u2net_finetune import prepare
from u2net_finetune import U2NetCallback


src_folder = "raw"
src_rgba_folder = "rgba"
tgt_folder = "u2net_finetune"
label_folder = "u2net_finetune_labels"


@AlphaSegmentationCallback.register("cascade_u2net")
class CascadeU2NetCallback(U2NetCallback):
    pass


@cflearn.LossProtocol.register("cascade_u2net")
class CascadeU2NetLoss(cflearn.LossProtocol):
    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        losses = {
            f"iou{i}": 1.0 - iou(pred, labels) for i, pred in enumerate(predictions)
        }
        losses[LOSS_KEY] = sum(losses.values())
        return losses


if __name__ == "__main__":
    prepare()
    train_loader, valid_loader = cflearn.cv.get_image_folder_loaders(
        tgt_folder,
        batch_size=16,
        num_workers=4,
        transform="for_salient_object_detection",
    )
    m = cflearn.cv.CarefreePipeline(
        "cascade_u2net",
        {
            "in_channels": 3,
            "out_channels": 1,
            "lv1_model_ckpt_path": "pretrained/model_lite_finetuned.pt",
            "lite": True,
        },
        loss_name="cascade_u2net",
        lr=1.0e-4,
        optimizer_name="sgd",
        scheduler_name="none",
        optimizer_config={
            "momentum": 0.9,
            "weight_decay": 5.0e-4,
            "nesterov": True,
        },
        loss_metrics_weights={"iou0": 1.0},
    )
    m.fit(train_loader, valid_loader, cuda="6")
