# type: ignore

import os
import cv2
import cflearn

import numpy as np
import torch.nn as nn

from typing import Any
from typing import List
from typing import Optional
from cflearn.types import losses_type
from cflearn.types import tensor_dict_type
from cflearn.constants import LOSS_KEY
from cflearn.constants import INPUT_KEY
from cflearn.constants import LABEL_KEY
from cflearn.constants import PREDICTIONS_KEY
from cflearn.protocol import TrainerState
from cflearn.api.cv import AlphaSegmentationCallback
from cflearn.misc.toolkit import iou
from cflearn.misc.toolkit import to_device
from cflearn.misc.toolkit import eval_context
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


@AlphaSegmentationCallback.register("u2net")
class U2NetCallback(AlphaSegmentationCallback):
    key = "images"

    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        with eval_context(trainer.model):
            logits = trainer.model.generate_from(batch[INPUT_KEY])
        self._save_seg_results(trainer, batch, logits)


@cflearn.LossProtocol.register("u2net")
class U2NetLoss(cflearn.LossProtocol):
    def _init_config(self) -> None:
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

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
            f"lv{i}": self.bce(pred, labels).mean((1, 2, 3))
            for i, pred in enumerate(predictions)
        }
        for i, pred in enumerate(predictions):
            losses[f"iou{i}"] = 1.0 - iou(pred, labels)
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
        "u2net",
        {
            "in_channels": 3,
            "out_channels": 1,
            "lite": True,
        },
        loss_name="u2net",
        loss_metrics_weights={"iou0": 1.0},
        # lr=4.0e-3,
        scheduler_name="none",
        finetune_config={
            # "pretrained_ckpt": "pretrained/model.pt",
            "pretrained_ckpt": "pretrained/model_lite.pt",
            # "freeze_except": r"(.*\.side_blocks\..*|.*\.out\..*)",
        },
    )
    m.fit(train_loader, valid_loader, cuda="1")
    # m.ddp(train_loader, valid_loader, cuda_list=[0, 2, 3, 4])
