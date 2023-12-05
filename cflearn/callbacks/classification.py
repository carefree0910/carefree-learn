import os

from cftool.cv import save_images
from cftool.array import to_device

from .general import ImageCallback
from ..schema import ITrainer
from ..toolkit import np_batch_to_tensor
from ..toolkit import make_indices_visualization_map
from ..constants import INPUT_KEY
from ..constants import PREDICTIONS_KEY


@ImageCallback.register("cv_clf")
class ImageClassificationCallback(ImageCallback):
    def log_artifacts(self, trainer: ITrainer) -> None:
        if not self.is_local_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = np_batch_to_tensor(batch)
        batch = to_device(batch, trainer.device)
        original = batch[INPUT_KEY]
        with trainer.model.eval_context():
            logits = trainer.model.m(original)[PREDICTIONS_KEY]
            labels_map = make_indices_visualization_map(logits.argmax(1))
        image_folder = self._prepare_folder(trainer)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(labels_map, os.path.join(image_folder, "labels.png"))


__all__ = [
    "ImageClassificationCallback",
]
