import os
import torch
import cflearn

from tqdm import tqdm
from cflearn.constants import INPUT_KEY
from cflearn.constants import INFO_PREFIX
from cflearn.misc.toolkit import to_device
from cflearn.misc.toolkit import save_images
from cflearn.misc.toolkit import eval_context
from cflearn.misc.toolkit import get_latest_workplace

cuda = "1"
log_folder = ".vq_vae"
code_export_folder = ".vq_vae_codes"


# export vq vae codes
def export_code_indices() -> None:
    os.makedirs(code_export_folder, exist_ok=True)
    train, valid = cflearn.cv.get_mnist(shuffle=False, transform="for_generation")
    for name, loader in zip(["train", "valid"], [train, valid]):
        code_indices = []
        for batch in tqdm(loader, desc=f"{name} codes", total=len(loader)):
            net = batch[INPUT_KEY].to(vq_vae.device)
            code_indices.append(vq_vae.get_code_indices(net).unsqueeze(1))
        all_codes = torch.cat(code_indices, dim=0)
        path = os.path.join(code_export_folder, f"{name}.pt")
        print(f"{INFO_PREFIX}saving {name} codes ({all_codes.shape})")
        torch.save(all_codes, path)


m_base = cflearn.cv.SimplePipeline
workplace = get_latest_workplace(log_folder)
vq_vae = m_base.load(m_base.pack(workplace), cuda=cuda).model


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
        with eval_context(trainer.model):
            sampled_indices = trainer.model.sample(batch_size, img_size)
        with eval_context(vq_vae):
            original = vq_vae.reconstruct_from(original_indices.squeeze(1))
            sampled = vq_vae.reconstruct_from(sampled_indices.squeeze(1))
        image_folder = self._prepare_folder(trainer)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(sampled, os.path.join(image_folder, "sampled.png"))


if __name__ == "__main__":
    export_code_indices()
    x_train = torch.load(os.path.join(code_export_folder, "train.pt"))
    x_valid = torch.load(os.path.join(code_export_folder, "valid.pt"))
    train_loader, valid_loader = cflearn.cv.get_tensor_loaders(
        x_train,
        y_train=x_train,
        x_valid=x_valid,
        y_valid=x_valid,
    )

    m = cflearn.cv.CarefreePipeline(
        "pixel_cnn",
        {"in_channels": 1, "num_classes": 16, "need_embedding": True},
        loss_name="cross_entropy",
        metric_names="acc",
    )
    m.fit(train_loader, valid_loader, cuda=cuda)
