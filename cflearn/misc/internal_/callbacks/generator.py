import os
import torch

import numpy as np

from .general import TrainerCallback
from .general import ArtifactCallback
from ...toolkit import to_numpy
from ...toolkit import to_torch
from ...toolkit import to_device
from ...toolkit import save_images
from ...toolkit import eval_context
from ...toolkit import min_max_normalize
from ....types import tensor_dict_type
from ....trainer import Trainer
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....models.cv.protocol import GeneratorMixin


@TrainerCallback.register("generator")
class GeneratorCallback(ArtifactCallback):
    key = "images"
    num_interpolations = 16

    def log_artifacts(self, trainer: Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[INPUT_KEY]
        model = trainer.model
        if not isinstance(model, GeneratorMixin):
            msg = "`GeneratorCallback` is only compatible with `GeneratorMixin`"
            raise ValueError(msg)
        is_conditional = model.is_conditional
        labels = None if not is_conditional else batch[LABEL_KEY]
        image_folder = self._prepare_folder(trainer)
        # original
        save_images(original, os.path.join(image_folder, "original.png"))
        # reconstruct
        if model.can_reconstruct:
            with eval_context(model):
                reconstructed = model.reconstruct(original, labels=labels)
            save_images(reconstructed, os.path.join(image_folder, "reconstructed.png"))
        # sample
        with eval_context(model):
            sampled = model.sample(len(original))
        save_images(sampled, os.path.join(image_folder, "sampled.png"))
        # interpolation
        with eval_context(model):
            interpolations = model.interpolate(self.num_interpolations)
        save_images(interpolations, os.path.join(image_folder, "interpolations.png"))
        # conditional sampling
        if model.num_classes is None:
            return None
        cond_folder = os.path.join(image_folder, "conditional")
        os.makedirs(cond_folder, exist_ok=True)
        with eval_context(model):
            for i in range(model.num_classes):
                sampled = model.sample(len(original), class_idx=i)
                interpolations = model.interpolate(len(original), class_idx=i)
                save_images(sampled, os.path.join(cond_folder, f"sampled_{i}.png"))
                path = os.path.join(cond_folder, f"interpolations_{i}.png")
                save_images(interpolations, path)


@TrainerCallback.register("sized_generator")
class SizedGeneratorCallback(GeneratorCallback):
    def log_artifacts(self, trainer: Trainer) -> None:
        if not self.is_rank_0:
            return None
        super().log_artifacts(trainer)
        image_folder = self._prepare_folder(trainer, check_num_keep=False)
        sample_method = getattr(trainer.model, "sample", None)
        if sample_method is None:
            raise ValueError(
                "`sample` should be implemented when `SizedGeneratorCallback` is used "
                "(and the `sample` method should support accepting `size` kwarg)"
            )
        with eval_context(trainer.model):
            for size in [32, 64, 128]:
                sampled = sample_method(4, size=size).cpu()
                path = os.path.join(image_folder, f"sampled_{size}x{size}.png")
                save_images(sampled, path)


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
    "GeneratorCallback",
    "SizedGeneratorCallback",
    "AlphaSegmentationCallback",
]
