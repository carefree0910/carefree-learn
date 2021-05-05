# type: ignore

import os
import cflearn

from torchvision.transforms import transforms
from cflearn.misc.toolkit import to_device
from cflearn.misc.toolkit import save_images
from cflearn.misc.toolkit import eval_context


@cflearn.ArtifactCallback.register("gan")
class GANCallback(cflearn.ArtifactCallback):
    key = "images"

    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[cflearn.INPUT_KEY]
        with eval_context(trainer.model):
            sampled = trainer.model.sample(len(original), batch.get(cflearn.LABEL_KEY))
        image_folder = self._prepare_folder(trainer)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(sampled, os.path.join(image_folder, "sampled.png"))


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ]
)
train_loader, valid_loader = cflearn.cv.get_mnist(transform=transform)

m = cflearn.cv.CarefreePipeline(
    "gan",
    {"img_size": 28, "in_channels": 1},
    optimizer_settings={
        "g_parameters": {
            "optimizer": "adam",
            "scheduler": "warmup",
        },
        "d_parameters": {
            "optimizer": "adam",
            "scheduler": "warmup",
        },
    },
)
m.fit(train_loader, valid_loader, cuda="0")
