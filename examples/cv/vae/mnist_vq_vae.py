# type: ignore

import os
import math
import cflearn

from cflearn.misc.toolkit import to_device
from cflearn.misc.toolkit import save_images
from cflearn.misc.toolkit import eval_context
from cflearn.misc.toolkit import make_indices_visualization_map
from cflearn.modules.blocks import upscale


@cflearn.ArtifactCallback.register("vq_vae")
class VQVAECallback(cflearn.ArtifactCallback):
    key = "images"

    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[cflearn.INPUT_KEY]
        model = trainer.model
        state = trainer.state
        with eval_context(model):
            outputs = model(0, batch, state)
        reconstructed = outputs[cflearn.PREDICTIONS_KEY]
        image_folder = self._prepare_folder(trainer)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(reconstructed, os.path.join(image_folder, "reconstructed.png"))
        with eval_context(model):
            codes, indices = model.sample_codebook(num_samples=len(original))
        save_images(codes, os.path.join(image_folder, "codes.png"))
        indices_map = make_indices_visualization_map(indices)
        save_images(indices_map, os.path.join(image_folder, "code_indices.png"))
        if num_classes is not None:
            with eval_context(model):
                for i in range(num_classes):
                    kwargs = dict(num_samples=len(original), class_idx=i)
                    codes, indices = model.sample_codebook(**kwargs)
                    save_images(codes, os.path.join(image_folder, f"codes_{i}.png"))
                    indices_map = make_indices_visualization_map(indices)
                    save_images(indices_map, os.path.join(image_folder, f"code_indices_{i}.png"))
        # inspect
        sample = reconstructed[:1]
        sample_indices = outputs["indices"][0].view(-1)
        sample_map = make_indices_visualization_map(sample_indices)
        with eval_context(model):
            sample_vis = model.sample_codebook(code_indices=sample_indices)[0]
        scaled = upscale(sample, math.sqrt(len(sample_indices)))
        save_images(scaled, os.path.join(image_folder, "sampled.png"))
        save_images(sample_map, os.path.join(image_folder, "sampled_idx.png"))
        save_images(sample_vis, os.path.join(image_folder, "sampled_codes.png"))


num_classes = 10
train_loader, valid_loader = cflearn.cv.get_mnist(transform="for_generation")

m = cflearn.cv.CarefreePipeline(
    "vq_vae",
    {
        "img_size": 28,
        "num_code": 16,
        "in_channels": 1,
        "target_downsample": 2,
        "num_classes": num_classes,
    },
)
m.fit(train_loader, valid_loader, cuda="1")
