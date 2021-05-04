# type: ignore

import os
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
    img, _ = batch
    img_label = (img * 255).long()
    return {cflearn.INPUT_KEY: img, cflearn.LABEL_KEY: img_label}


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


data_base = cflearn.data_dict["dl"]
loader_base = cflearn.loader_dict["dl"]

transform = transforms.ToTensor()

train_data = data_base(MNIST("data", transform=transform, download=True))
valid_data = data_base(MNIST("data", train=False, transform=transform, download=True))

train_pt_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # type: ignore
valid_pt_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # type: ignore

train_loader = loader_base(train_pt_loader, batch_callback)
valid_loader = loader_base(valid_pt_loader, batch_callback)

m = cflearn.cv.CarefreePipeline(
    "pixel_cnn",
    {"in_channels": 1, "num_classes": 256},
    loss_name="cross_entropy",
    metric_names="acc",
)
m.fit(train_loader, valid_loader, cuda="0")
