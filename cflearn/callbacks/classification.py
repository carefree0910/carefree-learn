import os

from cftool.cv import save_images
from cftool.array import to_device

from .general import ImageCallback
from ..misc.toolkit import eval_context
from ..misc.toolkit import make_indices_visualization_map
from ..schema import ITrainer
from ..constants import INPUT_KEY
from ..constants import PREDICTIONS_KEY


@ImageCallback.register("clf")
class ClassificationCallback(ImageCallback):
    def log_artifacts(self, trainer: ITrainer) -> None:
        if not self.is_local_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[INPUT_KEY]
        with eval_context(trainer.model):
            logits = trainer.model.classify(original)[PREDICTIONS_KEY]
            labels_map = make_indices_visualization_map(logits.argmax(1))
        image_folder = self._prepare_folder(trainer)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(labels_map, os.path.join(image_folder, "labels.png"))


__all__ = [
    "ClassificationCallback",
]
