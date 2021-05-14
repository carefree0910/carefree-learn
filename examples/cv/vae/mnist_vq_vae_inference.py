# type: ignore

import os
import torch
import cflearn

from tqdm import tqdm
from cflearn.constants import INPUT_KEY
from cflearn.constants import LABEL_KEY
from cflearn.constants import INFO_PREFIX
from cflearn.constants import ORIGINAL_LABEL_KEY
from cflearn.misc.toolkit import to_device
from cflearn.misc.toolkit import save_images
from cflearn.misc.toolkit import eval_context


cuda = "1"
log_folder = ".c_vq_vae"
inference_folder = ".c_vq_vae_inference"
code_export_folder = os.path.join(inference_folder, "codes")
num_classes = 10


# export vq vae codes
def export_code_indices() -> None:
    os.makedirs(code_export_folder, exist_ok=True)
    train, valid = cflearn.cv.get_mnist(shuffle=False, transform="for_generation")
    for name, loader in zip(["train", "valid"], [train, valid]):
        labels = []
        code_indices = []
        for batch in tqdm(loader, desc=f"{name} codes", total=len(loader)):
            labels.append(batch[LABEL_KEY])
            net = batch[INPUT_KEY].to(vq_vae.device)
            code_indices.append(vq_vae.get_code_indices(net).unsqueeze(1))
        all_codes = torch.cat(code_indices, dim=0)
        path = os.path.join(code_export_folder, f"{name}.pt")
        print(f"{INFO_PREFIX}saving {name} codes ({all_codes.shape})")
        torch.save(all_codes, path)
        all_labels = torch.cat(labels, dim=0)
        label_path = os.path.join(code_export_folder, f"{name}_labels.pt")
        print(f"{INFO_PREFIX}saving {name} labels ({all_labels.shape})")
        torch.save(all_labels, label_path)


m_base = cflearn.cv.SimplePipeline
vq_vae = m_base.load(m_base.pack(log_folder), cuda=cuda).model


# use pixel cnn to generate the codes
@cflearn.ArtifactCallback.register("pixel_cnn")
class PixelCNNCallback(cflearn.ArtifactCallback):
    key = "images"

    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original_indices = batch[cflearn.INPUT_KEY]
        img_size = original_indices.shape[2]
        batch_size = original_indices.shape[0]
        model = trainer.model
        with eval_context(model):
            sampled_indices = model.sample(batch_size, img_size)
        with eval_context(vq_vae):
            original = vq_vae.reconstruct_from(original_indices.squeeze(1))
            sampled = vq_vae.reconstruct_from(sampled_indices.squeeze(1))
        image_folder = self._prepare_folder(trainer)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(sampled, os.path.join(image_folder, "sampled.png"))
        if num_classes is not None:
            for i in range(num_classes):
                with eval_context(model):
                    sampled_indices = model.sample(batch_size, img_size, i)
                with eval_context(vq_vae):
                    sampled_indices = sampled_indices.squeeze(1)
                    sampled = vq_vae.reconstruct_from(sampled_indices, class_idx=i)
                save_images(sampled, os.path.join(image_folder, f"sampled_{i}.png"))


if __name__ == "__main__":
    export_code_indices()
    x_train = torch.load(os.path.join(code_export_folder, "train.pt"))
    y_train = torch.load(os.path.join(code_export_folder, "train_labels.pt"))
    x_valid = torch.load(os.path.join(code_export_folder, "valid.pt"))
    y_valid = torch.load(os.path.join(code_export_folder, "valid_labels.pt"))
    train_loader, valid_loader = cflearn.cv.get_tensor_loaders(
        x_train,
        y_train=x_train,
        x_valid=x_valid,
        y_valid=x_valid,
        train_others={ORIGINAL_LABEL_KEY: y_train},
        valid_others={ORIGINAL_LABEL_KEY: y_valid},
    )

    m = cflearn.cv.CarefreePipeline(
        "pixel_cnn",
        {
            "in_channels": 1,
            "num_classes": 16,
            "need_embedding": True,
            "num_conditional_classes": num_classes,
        },
        loss_name="cross_entropy",
        metric_names="acc",
        workplace=f"{inference_folder}/_logs",
    )
    m.fit(train_loader, valid_loader, cuda=cuda)
