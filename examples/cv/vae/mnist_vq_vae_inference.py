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


cuda = 1
log_folder = ".versions/c_vq_vae_v1"
inference_folder = ".versions/c_vq_vae_inference_v1"
code_export_folder = os.path.join(inference_folder, "codes")
num_classes = 10


# export vq vae codes
def export_code_indices() -> None:
    os.makedirs(code_export_folder, exist_ok=True)
    code_data = cflearn.cv.MNISTData(shuffle=False, transform="for_generation")
    code_data.prepare(None)
    code_train, code_valid = code_data.initialize()
    for name, loader in zip(["train", "valid"], [code_train, code_valid]):
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
vq_vae: cflearn.VQVAE = m_base.load(m_base.pack(log_folder), cuda=cuda).model


# use pixel cnn to generate the codes
@cflearn.ImageCallback.register("pixel_cnn")
class PixelCNNCallback(cflearn.ImageCallback):
    def __init__(self, num_keep: int = 25, num_interpolations: int = 16):
        super().__init__(num_keep)
        self.num_interpolations = num_interpolations

    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        device = trainer.device
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, device)
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
            conditional_folder = os.path.join(image_folder, "conditional")
            os.makedirs(conditional_folder, exist_ok=True)
            for i in range(num_classes):
                with eval_context(model):
                    sampled_indices = model.sample(batch_size, img_size, i)
                with eval_context(vq_vae):
                    sampled = vq_vae.reconstruct_from(sampled_indices, class_idx=i)
                sampled_path = os.path.join(conditional_folder, f"sampled_{i}.png")
                save_images(sampled, sampled_path)
                with eval_context(model):
                    i1 = model.sample(self.num_interpolations, img_size, i)
                    i2 = model.sample(self.num_interpolations, img_size, i)
                with eval_context(vq_vae):
                    z1, z2 = map(vq_vae.get_code, [i1, i2])
                    ratio = torch.linspace(
                        0.0,
                        1.0,
                        self.num_interpolations,
                        device=device,
                    )
                    ratio = ratio.view(-1, 1, 1, 1)
                    z_q = ratio * z1 + (1.0 - ratio) * z2
                    shape = [self.num_interpolations, 1]
                    labels = torch.full(shape, i, device=device)
                    interpolations = vq_vae.decode(z_q, labels=labels)
                interp_path = os.path.join(conditional_folder, f"interpolation_{i}.png")
                save_images(interpolations, interp_path)


if __name__ == "__main__":
    export_code_indices()
    x_train = torch.load(os.path.join(code_export_folder, "train.pt"))
    y_train = torch.load(os.path.join(code_export_folder, "train_labels.pt"))
    x_valid = torch.load(os.path.join(code_export_folder, "valid.pt"))
    y_valid = torch.load(os.path.join(code_export_folder, "valid_labels.pt"))
    data = cflearn.TensorData(
        x_train,
        y_train=x_train,
        x_valid=x_valid,
        y_valid=x_valid,
        train_others={ORIGINAL_LABEL_KEY: y_train},
        valid_others={ORIGINAL_LABEL_KEY: y_valid},
    )

    m = cflearn.api.pixel_cnn(
        16,
        model_config={
            "need_embedding": True,
            "num_conditional_classes": num_classes,
        },
        workplace=f"{inference_folder}/_logs",
    )
    m.fit(data, cuda=cuda)
