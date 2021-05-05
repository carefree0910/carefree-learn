# type: ignore

import os
import cflearn

from torchvision.transforms import transforms
from cflearn.misc.toolkit import to_device
from cflearn.misc.toolkit import save_images
from cflearn.misc.toolkit import eval_context
from cflearn.misc.toolkit import make_indices_visualization_map


@cflearn.ArtifactCallback.register("clf")
class ClassificationCallback(cflearn.ArtifactCallback):
    key = "images"

    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[cflearn.INPUT_KEY]
        with eval_context(trainer.model):
            logits = trainer.model.classify(original)
            labels_map = make_indices_visualization_map(logits.argmax(1))
        image_folder = self._prepare_folder(trainer)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(labels_map, os.path.join(image_folder, "labels.png"))


transform = transforms.ToTensor()
train_loader, valid_loader = cflearn.cv.get_mnist(transform=transform)

m = cflearn.cv.CarefreePipeline(
    "clf",
    {"img_size": 28, "in_channels": 1, "num_classes": 10},
    loss_name="cross_entropy",
    metric_names="acc",
)
m.fit(train_loader, valid_loader, cuda="0")
