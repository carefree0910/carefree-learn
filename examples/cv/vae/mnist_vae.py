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
    img, labels = batch
    return {cflearn.INPUT_KEY: img, cflearn.LABEL_KEY: labels.view(-1, 1)}


@cflearn.ArtifactCallback.register("vae")
class VAECallback(cflearn.ArtifactCallback):
    key = "images"

    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[cflearn.INPUT_KEY]
        with eval_context(trainer.model):
            reconstructed = trainer.model.reconstruct(original)
            sampled = trainer.model.sample(len(original))
        image_folder = self._prepare_folder(trainer)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(reconstructed, os.path.join(image_folder, "reconstructed.png"))
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

# loss = cflearn.loss_dict["vae"]()
# vae = cflearn.VanillaVAE(28, 1)
# inference = cflearn.DLInference(model=vae)
# cf_trainer = cflearn.Trainer(
#     workplace="_logs",
#     loss_metrics_weights={"kld": 0.002, "mse": 1.0},
#     callbacks=[VAECallback(), _LogMetricsMsgCallback()],
# )
# cf_trainer.fit(loss, vae, inference, train_loader, valid_loader, cuda="0")

m = cflearn.cv.CarefreePipeline(
    "vae",
    {"img_size": 28, "in_channels": 1},
)
m.fit(train_loader, valid_loader, cuda="0")
