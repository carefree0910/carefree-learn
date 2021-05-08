# type: ignore

import os
import cflearn

from cflearn.misc.toolkit import to_device
from cflearn.misc.toolkit import save_images
from cflearn.misc.toolkit import eval_context


@cflearn.ArtifactCallback.register("pixel_cnn")
class PixelCNNCallback(cflearn.ArtifactCallback):
    key = "images"

    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[cflearn.INPUT_KEY]
        with eval_context(trainer.model):
            sampled = trainer.model.sample(len(original), original.shape[2])
        image_folder = self._prepare_folder(trainer)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(sampled, os.path.join(image_folder, "sampled.png"))


train_loader, valid_loader = cflearn.cv.get_mnist(
    transform="for_classification",
    label_callback=lambda batch: (batch[0] * 255).long(),
)

m = cflearn.cv.CarefreePipeline(
    "pixel_cnn",
    {"in_channels": 1, "num_classes": 256},
    loss_name="cross_entropy",
    metric_names="acc",
)
m.fit(train_loader, valid_loader, cuda="0")
