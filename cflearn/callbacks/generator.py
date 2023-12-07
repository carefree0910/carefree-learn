import os

from cftool.cv import save_images

from .general import ImageCallback
from ..data import TensorBatcher
from ..schema import ITrainer
from ..schema import TrainerCallback
from ..modules import IGenerator
from ..toolkit import eval_context
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY


@TrainerCallback.register("ldm")
@TrainerCallback.register("ddpm")
@TrainerCallback.register("ae_kl")
@TrainerCallback.register("ae_vq")
@TrainerCallback.register("vae")
@TrainerCallback.register("gan")
class GeneratorCallback(ImageCallback):
    def __init__(self, num_keep: int = 25, num_interpolations: int = 16):
        super().__init__(num_keep)
        self.num_interpolations = num_interpolations

    def log_artifacts(self, trainer: ITrainer) -> None:
        if not self.is_local_rank_0:
            return None
        batch = TensorBatcher(trainer.validation_loader, trainer.device).get_one_batch()
        original = batch[INPUT_KEY]
        m = trainer.model.m
        if not isinstance(m, IGenerator):
            msg = "`GeneratorCallback` is only compatible with `IGenerator`"
            raise ValueError(msg)
        labels = None if not m.is_conditional else batch[LABEL_KEY]
        image_folder = self._prepare_folder(trainer)
        # original
        save_images(original, os.path.join(image_folder, "original.png"))
        # reconstruct
        with eval_context(m):
            recon = m.reconstruct(original, labels=labels)
            if recon is not None:
                save_images(recon, os.path.join(image_folder, "reconstructed.png"))
        # sample
        with eval_context(m):
            sampled = m.sample(len(original))
        save_images(sampled, os.path.join(image_folder, "sampled.png"))
        # interpolation
        with eval_context(m):
            interpolations = m.interpolate(self.num_interpolations)
        save_images(interpolations, os.path.join(image_folder, "interpolations.png"))
        # conditional sampling
        if m.num_classes is None:
            return None
        cond_folder = os.path.join(image_folder, "conditional")
        os.makedirs(cond_folder, exist_ok=True)
        with eval_context(m):
            for i in range(m.num_classes):
                sampled = m.sample(len(original), class_idx=i)
                interpolations = m.interpolate(len(original), class_idx=i)
                save_images(sampled, os.path.join(cond_folder, f"sampled_{i}.png"))
                path = os.path.join(cond_folder, f"interpolations_{i}.png")
                save_images(interpolations, path)


__all__ = [
    "GeneratorCallback",
]
