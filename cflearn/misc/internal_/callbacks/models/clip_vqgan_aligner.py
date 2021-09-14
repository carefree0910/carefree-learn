import os
import cv2
import torch

from PIL import Image
from tqdm import tqdm

from .....trainer import Trainer
from .....trainer import TrainerCallback
from .....misc.toolkit import eval_context


@TrainerCallback.register("clip_vqgan_aligner")
class CLIPWithVQGANAlignerCallback(TrainerCallback):
    def __init__(self, outputs_folder: str = "outputs", video_file: str = "video.mp4"):
        super().__init__()
        self.outputs_folder = outputs_folder
        self.video_file = video_file

    def before_loop(self, trainer: Trainer) -> None:
        self.log_artifacts(trainer)

    def log_artifacts(self, trainer: Trainer) -> None:
        if not self.is_rank_0:
            return None
        state = trainer.state
        if state.is_terminate:
            return None
        outputs_folder = os.path.join(trainer.workplace, self.outputs_folder)
        os.makedirs(outputs_folder, exist_ok=True)
        model = trainer.model
        with eval_context(model):
            img_tensor = model.generate()[0]
        img_tensor = (img_tensor * 255.0).to(torch.uint8)
        img = Image.fromarray(img_tensor.cpu().numpy().transpose([1, 2, 0]))
        img.save(os.path.join(outputs_folder, f"{state.step:06d}.png"))

    def finalize(self, trainer: Trainer) -> None:
        workplace = trainer.workplace
        src = os.path.join(workplace, self.outputs_folder)
        video_path = os.path.join(src, self.video_file)
        resolution = trainer.model.resolution
        fourcc = cv2.VideoWriter_fourcc("M", "P", "4", "V")
        out = cv2.VideoWriter(video_path, fourcc, 25, resolution, isColor=True)
        files = [file for file in os.listdir(src) if file.endswith(".png")]
        for file in tqdm(sorted(files)):
            img = cv2.imread(os.path.join(src, file))
            out.write(img)
        out.release()


__all__ = [
    "CLIPWithVQGANAlignerCallback",
]
