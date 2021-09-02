# type: ignore

import os
import cflearn
import argparse

from torchvision.transforms import transforms
from cflearn.misc.toolkit import to_device
from cflearn.misc.toolkit import save_images
from cflearn.misc.toolkit import eval_context

# CI
parser = argparse.ArgumentParser()
parser.add_argument("--ci", type=int, default=0)
args = parser.parse_args()
is_ci = bool(args.ci) or True


@cflearn.ImageCallback.register("pixel_cnn")
class PixelCNNCallback(cflearn.ImageCallback):
    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[cflearn.INPUT_KEY].float() / 255.0
        num_samples, img_size = len(original), original.shape[2]
        model = trainer.model
        with eval_context(model):
            sampled_indices = model.sample(num_samples, 2 if is_ci else img_size)
            sampled = sampled_indices.float() / 255.0
        image_folder = self._prepare_folder(trainer)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(sampled, os.path.join(image_folder, "sampled.png"))
        if num_conditional_classes is not None:
            with eval_context(model):
                for i in range(num_conditional_classes):
                    sampled_indices = model.sample(num_samples, img_size, i)
                    sampled = sampled_indices.float() / 255.0
                    save_images(sampled, os.path.join(image_folder, f"sampled_{i}.png"))


num_conditional_classes = None if is_ci else 10
data = cflearn.cv.MNISTData(
    root="../data",
    batch_size=4 if is_ci else 64,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 255).long()),
        ]
    ),
    label_callback=lambda batch: batch[0],
)

m = cflearn.cv.CarefreePipeline(
    "pixel_cnn",
    {
        "in_channels": 1,
        "num_classes": 256,
        "num_conditional_classes": num_conditional_classes,
    },
    loss_name="cross_entropy",
    metric_names="acc",
    fixed_steps=1 if is_ci else None,
    valid_portion=0.0001 if is_ci else 1.0,
)
m.fit(data, cuda=None if is_ci else 0)
