import os

from cftool.cv import save_images
from cftool.array import to_device

from .general import ImageCallback
from ..misc.toolkit import eval_context
from ..schema import ITrainer
from ..schema import TrainerCallback
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..models.schemas.cv import GeneratorMixin


@TrainerCallback.register("ldm")
@TrainerCallback.register("ddpm")
@TrainerCallback.register("ae_kl")
@TrainerCallback.register("ae_vq")
@TrainerCallback.register("gan")
@TrainerCallback.register("vae")
@TrainerCallback.register("vae2d")
@TrainerCallback.register("style_vae")
@TrainerCallback.register("generator")
class GeneratorCallback(ImageCallback):
    def __init__(self, num_keep: int = 25, num_interpolations: int = 16):
        super().__init__(num_keep)
        self.num_interpolations = num_interpolations

    def log_artifacts(self, trainer: ITrainer) -> None:
        if not self.is_local_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[INPUT_KEY]
        model = trainer.model
        if not isinstance(model, GeneratorMixin):
            msg = "`GeneratorCallback` is only compatible with `GeneratorMixin`"
            raise ValueError(msg)
        is_conditional = getattr(model, "is_conditional", False)
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
        if getattr(model, "num_classes", None) is None:
            return None
        cond_folder = os.path.join(image_folder, "conditional")
        os.makedirs(cond_folder, exist_ok=True)
        with eval_context(model):
            for i in range(model.num_classes or 0):
                sampled = model.sample(len(original), class_idx=i)
                interpolations = model.interpolate(len(original), class_idx=i)
                save_images(sampled, os.path.join(cond_folder, f"sampled_{i}.png"))
                path = os.path.join(cond_folder, f"interpolations_{i}.png")
                save_images(interpolations, path)


@TrainerCallback.register("siren_gan")
@TrainerCallback.register("siren_vae")
@TrainerCallback.register("sized_generator")
class SizedGeneratorCallback(GeneratorCallback):
    def log_artifacts(self, trainer: ITrainer) -> None:
        if not self.is_local_rank_0:
            return None
        super().log_artifacts(trainer)
        model = trainer.model
        image_folder = self._prepare_folder(trainer, check_num_keep=False)
        sample_method = getattr(model, "sample", None)
        if sample_method is None:
            raise ValueError(
                "`sample` should be implemented when `SizedGeneratorCallback` is used "
                "(and the `sample` method should support accepting `size` kwarg)"
            )
        resolution = getattr(model, "img_size", 32)
        with eval_context(model):
            for i in range(1, 4):
                size = resolution * 2**i
                batch_size = 1 if size > 256 else 4
                sampled = sample_method(batch_size, size=size).cpu()
                path = os.path.join(image_folder, f"sampled_{size}x{size}.png")
                save_images(sampled, path)


__all__ = [
    "GeneratorCallback",
    "SizedGeneratorCallback",
]
