from ..generator import AlphaSegmentationCallback
from .....trainer import Trainer
from .....constants import INPUT_KEY
from .....misc.toolkit import to_device
from .....misc.toolkit import eval_context


@AlphaSegmentationCallback.register("u2net")
class U2NetCallback(AlphaSegmentationCallback):
    key = "images"

    def log_artifacts(self, trainer: Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        with eval_context(trainer.model):
            logits = trainer.model.generate_from(batch[INPUT_KEY])
        self._save_seg_results(trainer, batch, logits)


__all__ = [
    "U2NetCallback",
]
