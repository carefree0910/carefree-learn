# type: ignore

import os
import cflearn

from cflearn.misc.toolkit import save_images
from cflearn.misc.toolkit import eval_context


@cflearn.GeneratorCallback.register("siren_vae")
class SirenVAECallback(cflearn.GeneratorCallback):
    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        super().log_artifacts(trainer)
        image_folder = self._prepare_folder(trainer)
        with eval_context(trainer.model):
            for size in [64, 128, 256]:
                sampled = trainer.model.sample(4, size=size)
                path = os.path.join(image_folder, f"sampled_{size}x{size}.png")
                save_images(sampled, path)


num_classes = 10
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
