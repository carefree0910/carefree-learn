import os

from typing import Any
from typing import Dict
from typing import Optional
from cftool.cv import save_images
from cftool.misc import shallow_copy_dict

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
    def __init__(
        self,
        num_keep: int = 25,
        num_interpolations: int = 16,
        *,
        sample_kwargs: Optional[Dict[str, Any]] = None,
        reconstruct_kwargs: Optional[Dict[str, Any]] = None,
        interpolate_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(num_keep)
        self.num_interpolations = num_interpolations
        self.sample_kwargs = sample_kwargs or {}
        self.reconstruct_kwargs = reconstruct_kwargs or {}
        self.interpolate_kwargs = interpolate_kwargs or {}

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
        sample_kw = shallow_copy_dict(self.sample_kwargs)
        reconstruct_kw = shallow_copy_dict(self.reconstruct_kwargs)
        interpolate_kw = shallow_copy_dict(self.interpolate_kwargs)
        # original
        save_images(original, os.path.join(image_folder, "original.png"))
        # reconstruct
        with eval_context(m):
            recon = m.reconstruct(original, labels=labels, kwargs=reconstruct_kw)
            if recon is not None:
                save_images(recon, os.path.join(image_folder, "reconstructed.png"))
        # sample
        with eval_context(m):
            sampled = m.sample(len(original), kwargs=shallow_copy_dict(sample_kw))
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
                num = len(original)
                i_sample_kw = shallow_copy_dict(sample_kw)
                i_interp_kw = shallow_copy_dict(interpolate_kw)
                sampled = m.sample(num, class_idx=i, kwargs=i_sample_kw)
                interpolations = m.interpolate(num, class_idx=i, kwargs=i_interp_kw)
                save_images(sampled, os.path.join(cond_folder, f"sampled_{i}.png"))
                path = os.path.join(cond_folder, f"interpolations_{i}.png")
                save_images(interpolations, path)


__all__ = [
    "GeneratorCallback",
]
