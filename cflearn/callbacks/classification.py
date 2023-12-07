import os

from cftool.cv import save_images

from .general import ImageCallback
from ..data import TensorBatcher
from ..schema import ITrainer
from ..toolkit import make_indices_visualization_map
from ..constants import INPUT_KEY
from ..constants import PREDICTIONS_KEY


@ImageCallback.register("cv_clf")
class ImageClassificationCallback(ImageCallback):
    def log_artifacts(self, trainer: ITrainer) -> None:
        if not self.is_local_rank_0:
            return None
        batch = TensorBatcher(trainer.validation_loader, trainer.device).get_one_batch()
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
