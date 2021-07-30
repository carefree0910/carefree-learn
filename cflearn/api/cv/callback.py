import os
import torch

import numpy as np

from ...types import tensor_dict_type
from ...trainer import Trainer
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY
from ...misc.toolkit import to_numpy
from ...misc.toolkit import to_torch
from ...misc.toolkit import save_images
from ...misc.toolkit import min_max_normalize
from ...misc.internal_.callback import ArtifactCallback


class AlphaSegmentationCallback(ArtifactCallback):
    key = "images"

    def _save_seg_results(
        self,
        trainer: Trainer,
        batch: tensor_dict_type,
        logits: torch.Tensor,
    ) -> None:
        original = batch[INPUT_KEY]
        label = batch[LABEL_KEY].float()
        seg_map = min_max_normalize(torch.sigmoid(logits))
        sharp_map = (seg_map > 0.5).to(torch.float32)
        image_folder = self._prepare_folder(trainer)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(label, os.path.join(image_folder, "label.png"))
        save_images(seg_map, os.path.join(image_folder, "mask.png"))
        save_images(sharp_map, os.path.join(image_folder, "mask_sharp.png"))
        np_original = min_max_normalize(to_numpy(original))
        np_label, np_mask, np_sharp = map(to_numpy, [label, seg_map, sharp_map])
        rgba = np.concatenate([np_original, np_label], axis=1)
        rgba_pred = np.concatenate([np_original, np_mask], axis=1)
        rgba_sharp = np.concatenate([np_original, np_sharp], axis=1)
        save_images(to_torch(rgba), os.path.join(image_folder, "rgba.png"))
        save_images(to_torch(rgba_pred), os.path.join(image_folder, "rgba_pred.png"))
        save_images(to_torch(rgba_sharp), os.path.join(image_folder, "rgba_sharp.png"))


__all__ = [
    "AlphaSegmentationCallback",
]
