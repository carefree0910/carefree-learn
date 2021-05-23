# type: ignore

import os
import cflearn

from cflearn.misc.toolkit import to_device
from cflearn.misc.toolkit import save_images
from cflearn.misc.toolkit import eval_context


@cflearn.ArtifactCallback.register("siren_vae")
class SirenVAECallback(cflearn.ArtifactCallback):
    key = "images"

    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[cflearn.INPUT_KEY]
        labels = None if num_classes is None else batch[cflearn.LABEL_KEY]
        with eval_context(trainer.model):
            reconstructed = trainer.model.reconstruct(original, labels=labels)
            sampled = trainer.model.sample(len(original))
        image_folder = self._prepare_folder(trainer)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(reconstructed, os.path.join(image_folder, "reconstructed.png"))
        save_images(sampled, os.path.join(image_folder, "sampled.png"))
        # super resolution sampling
        with eval_context(trainer.model):
            for size in [64, 128, 256]:
                sampled = trainer.model.sample(4, size=size)
                path = os.path.join(image_folder, f"sampled_{size}x{size}.png")
                save_images(sampled, path)
        # conditional sampling
        if num_classes is None:
            return None
        with eval_context(trainer.model):
            for i in range(num_classes):
                sampled = trainer.model.sample(len(original), class_idx=i)
                save_images(sampled, os.path.join(image_folder, f"sampled_{i}.png"))


num_classes = None
train_loader, valid_loader = cflearn.cv.get_mnist(transform="for_generation")

m = cflearn.cv.CarefreePipeline(
    "siren_vae",
    {
        "img_size": 28,
        "in_channels": 1,
        "num_classes": num_classes,
    },
    loss_name="vae",
    loss_metrics_weights={"kld": 0.001, "mse": 1.0},
)
m.fit(train_loader, valid_loader, cuda="3")
