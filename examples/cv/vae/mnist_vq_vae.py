# type: ignore

import os
import math
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
from cflearn.modules.blocks import upscale


def batch_callback(batch: Tuple[Tensor, Tensor]) -> tensor_dict_type:
    img, labels = batch
    return {cflearn.INPUT_KEY: img, cflearn.LABEL_KEY: labels.view(-1, 1)}


@cflearn.TrainerCallback.register("vq_vae")
class VQVAECallback(cflearn.TrainerCallback):
    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[cflearn.INPUT_KEY]
        state = trainer.state
        with eval_context(trainer.model):
            outputs = trainer.model(0, batch, state)
            codes, indices = trainer.model.sample_codebook(num_samples=len(original))
        reconstructed = outputs[cflearn.PREDICTIONS_KEY]
        image_folder = os.path.join(trainer.workplace, "images", str(state.step))
        os.makedirs(image_folder, exist_ok=True)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(reconstructed, os.path.join(image_folder, f"reconstructed.png"))
        save_images(codes, os.path.join(image_folder, "codes.png"))
        indices_map = trainer.model.make_map_from(indices)
        save_images(indices_map, os.path.join(image_folder, "code_indices.png"))
        # inspect
        sample = reconstructed[:1]
        sample_indices = outputs["indices"][0].view(-1)
        sample_indices_map = trainer.model.make_map_from(sample_indices)
        sample_indices_vis = trainer.model.sample_codebook(indices=sample_indices)[0]
        scaled = upscale(sample, math.sqrt(len(sample_indices)))
        save_images(scaled, os.path.join(image_folder, "sampled.png"))
        save_images(sample_indices_map, os.path.join(image_folder, "sampled_idx.png"))
        save_images(sample_indices_vis, os.path.join(image_folder, "sampled_codes.png"))


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
    "vq_vae",
    {
        "img_size": 28,
        "num_code": 16,
        "in_channels": 1,
        "target_downsample": 2,
    },
    loss_name="vq_vae",
    callback_names=["vq_vae", "_log_metrics_msg"],
)
m.fit(train_loader, valid_loader, cuda="0")
