# type: ignore

import os
import shutil

import cflearn

from torch import Tensor
from typing import Tuple
from cflearn.types import tensor_dict_type
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from cflearn.misc.toolkit import to_device
from cflearn.misc.toolkit import save_images
from cflearn.misc.toolkit import eval_context


def batch_callback(batch: Tuple[Tensor, Tensor]) -> tensor_dict_type:
    img, labels = batch
    return {cflearn.INPUT_KEY: img, cflearn.LABEL_KEY: labels.view(-1, 1)}


@cflearn.TrainerCallback.register("gan")
class GANCallback(cflearn.TrainerCallback):
    def __init__(self, num_keep: int = 25):
        super().__init__()
        self.num_keep = num_keep

    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[cflearn.INPUT_KEY]
        with eval_context(trainer.model):
            sampled = trainer.model.sample(len(original), batch.get(cflearn.LABEL_KEY))
        image_folder = os.path.join(trainer.workplace, "images")
        os.makedirs(image_folder, exist_ok=True)
        current_steps = sorted(map(int, os.listdir(image_folder)))
        if len(current_steps) >= self.num_keep:
            for step in current_steps[: -self.num_keep]:
                shutil.rmtree(os.path.join(image_folder, str(step)))
        image_folder = os.path.join(image_folder, str(trainer.state.step))
        os.makedirs(image_folder, exist_ok=True)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(sampled, os.path.join(image_folder, "sampled.png"))


data_base = cflearn.data_dict["dl"]
loader_base = cflearn.loader_dict["dl"]

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ]
)

train_data = data_base(MNIST("data", transform=transform, download=True))
valid_data = data_base(MNIST("data", train=False, transform=transform, download=True))

train_pt_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # type: ignore
valid_pt_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # type: ignore

train_loader = loader_base(train_pt_loader, batch_callback)
valid_loader = loader_base(valid_pt_loader, batch_callback)

m = cflearn.cv.SimplePipeline(
    "gan",
    {"img_size": 28, "in_channels": 1},
    loss_name="mse",
    monitor_names=[],
    callback_names="gan",
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
