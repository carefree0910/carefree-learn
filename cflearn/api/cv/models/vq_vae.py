import os
import torch

from tqdm import tqdm
from typing import Any
from typing import Optional
from cftool.misc import print_info
from cftool.misc import random_hash
from cftool.array import to_device
from cftool.array import save_images

from ...api import load
from ...api import pack
from ...api import pixel_cnn
from ....data import TensorData
from ....data import CVDataModule
from ....schema import ITrainer
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import ORIGINAL_LABEL_KEY
from ....misc.toolkit import eval_context
from ....misc.internal_.callbacks import ImageCallback
from ....models.cv.vae.vector_quantized import VQVAE


def register_callback(vqvae: VQVAE, num_classes: Optional[int]) -> str:
    tmp_name = random_hash()

    @ImageCallback.register(tmp_name)
    class _(ImageCallback):
        def __init__(self, num_keep: int = 25, num_interpolations: int = 16):
            super().__init__(num_keep)
            self.num_interpolations = num_interpolations

        def log_artifacts(self, trainer: ITrainer) -> None:
            if not self.is_rank_0:
                return None
            device = trainer.device
            batch = next(iter(trainer.validation_loader))
            batch = to_device(batch, device)
            original_indices = batch[INPUT_KEY]
            img_size = original_indices.shape[2]
            batch_size = original_indices.shape[0]
            model = trainer.model.core
            with eval_context(model):
                sampled_indices = model.sample(batch_size, img_size)
            with eval_context(vqvae):
                original = vqvae.reconstruct_from(original_indices.squeeze(1))
                sampled = vqvae.reconstruct_from(sampled_indices.squeeze(1))
            image_folder = self._prepare_folder(trainer)
            save_images(original, os.path.join(image_folder, "original.png"))
            save_images(sampled, os.path.join(image_folder, "sampled.png"))
            if num_classes is not None:
                conditional_folder = os.path.join(image_folder, "conditional")
                os.makedirs(conditional_folder, exist_ok=True)
                for i in range(num_classes):
                    with eval_context(model):
                        sampled_indices = model.sample(batch_size, img_size, i)
                    with eval_context(vqvae):
                        sampled = vqvae.reconstruct_from(sampled_indices, class_idx=i)
                    sampled_path = os.path.join(conditional_folder, f"sampled_{i}.png")
                    save_images(sampled, sampled_path)
                    with eval_context(model):
                        i1 = model.sample(self.num_interpolations, img_size, i)
                        i2 = model.sample(self.num_interpolations, img_size, i)
                    with eval_context(vqvae):
                        z1, z2 = map(vqvae.get_code, [i1, i2])
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
                        interpolations = vqvae.decode(z_q, labels=labels)
                    save_images(
                        interpolations,
                        os.path.join(conditional_folder, f"interpolation_{i}.png"),
                    )

    return tmp_name


class VQVAEInference:
    tmp_callback_name: Optional[str] = None

    def __init__(
        self,
        workplace: str,
        *,
        num_codes: int,
        vqvae_log_folder: str,
        num_classes: Optional[int] = None,
        inference_model: str = "pixel_cnn",
        cuda: Optional[int] = None,
        **kwargs: Any,
    ):
        self.cuda = cuda
        self.debug = kwargs.get("debug", False)
        self.vqvae = load(pack(vqvae_log_folder), cuda=cuda).model.core
        self.code_export_folder = os.path.join(workplace, "codes")
        if VQVAEInference.tmp_callback_name is not None:
            ImageCallback.remove(VQVAEInference.tmp_callback_name)
        VQVAEInference.tmp_callback_name = register_callback(self.vqvae, num_classes)
        if inference_model == "pixel_cnn":
            model_config = kwargs.setdefault("model_config", {})
            model_config["need_embedding"] = True
            model_config["num_conditional_classes"] = num_classes
            kwargs.setdefault("workplace", os.path.join(workplace, "_logs"))
            kwargs.setdefault("callback_names", VQVAEInference.tmp_callback_name)
            self.m = pixel_cnn(num_codes, **kwargs)
        else:
            msg = f"inference model '{inference_model}' is not implemented"
            raise NotImplementedError(msg)

    def export_code_indices(self, data: CVDataModule, export_folder: str) -> None:
        os.makedirs(export_folder, exist_ok=True)
        finished_path = os.path.join(export_folder, "__finished__")
        if os.path.isfile(finished_path):
            return None
        code_train, code_valid = data.get_loaders()
        for name, loader in zip(["train", "valid"], [code_train, code_valid]):
            if loader is None:
                continue
            labels = []
            code_indices = []
            for batch in tqdm(loader, desc=f"{name} codes", total=len(loader)):
                labels.append(batch[LABEL_KEY])
                net = batch[INPUT_KEY].to(self.vqvae.device)
                code_indices.append(self.vqvae.get_code_indices(net).unsqueeze(1))
                if self.debug:
                    break
            all_codes = torch.cat(code_indices, dim=0)
            path = os.path.join(export_folder, f"{name}.pt")
            print_info(f"saving {name} codes ({all_codes.shape})")
            torch.save(all_codes, path)
            all_labels = torch.cat(labels, dim=0)
            label_path = os.path.join(export_folder, f"{name}_labels.pt")
            print_info(f"saving {name} labels ({all_labels.shape})")
            torch.save(all_labels, label_path)
        if not self.debug:
            with open(finished_path, "w"):
                pass

    def fit(self, data: CVDataModule) -> "VQVAEInference":
        export_folder = self.code_export_folder
        self.export_code_indices(data, export_folder)
        x_train = torch.load(os.path.join(export_folder, "train.pt"))
        y_train = torch.load(os.path.join(export_folder, "train_labels.pt"))
        x_valid = torch.load(os.path.join(export_folder, "valid.pt"))
        y_valid = torch.load(os.path.join(export_folder, "valid_labels.pt"))
        tensor_data = TensorData(
            x_train,
            y_train=x_train,
            x_valid=x_valid,
            y_valid=x_valid,
            train_others={ORIGINAL_LABEL_KEY: y_train},
            valid_others={ORIGINAL_LABEL_KEY: y_valid},
        )
        self.m.fit(tensor_data, cuda=self.cuda)
        return self


__all__ = [
    "VQVAEInference",
]
