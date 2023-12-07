import os
import torch

from tqdm import tqdm
from typing import Optional
from cftool.cv import save_images
from cftool.misc import print_info
from cftool.misc import random_hash

from ..api import pack
from ..api import load_inference
from ...data import ArrayData
from ...data import TensorBatcher
from ...schema import device_type
from ...schema import IData
from ...schema import ITrainer
from ...schema import DLConfig
from ...schema import DataConfig
from ...toolkit import get_torch_device
from ...toolkit import eval_context
from ...modules import VQVAE
from ...modules import IAutoRegressor
from ...pipeline import DLTrainingPipeline
from ...callbacks import ImageCallback
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY


def register_callback(vqvae: VQVAE, num_classes: Optional[int]) -> str:
    tmp_name = random_hash()

    @ImageCallback.register(tmp_name)
    class _(ImageCallback):
        def __init__(self, num_keep: int = 25, num_interpolations: int = 16):
            super().__init__(num_keep)
            self.num_interpolations = num_interpolations

        def log_artifacts(self, trainer: ITrainer) -> None:
            if not self.is_local_rank_0:
                return None
            device = trainer.device
            batch = TensorBatcher(trainer.validation_loader, device).get_one_batch()
            original_indices = batch[INPUT_KEY]
            labels = batch[LABEL_KEY]
            img_size = original_indices.shape[2]
            batch_size = original_indices.shape[0]
            m: IAutoRegressor = trainer.model.m
            with eval_context(m):
                sampled_indices = m.sample(batch_size, labels=labels, img_size=img_size)
            with eval_context(vqvae):
                original = vqvae.reconstruct_from(original_indices, labels=labels)
                sampled = vqvae.reconstruct_from(sampled_indices, labels=labels)
            image_folder = self._prepare_folder(trainer)
            save_images(original, os.path.join(image_folder, "original.png"))
            save_images(sampled, os.path.join(image_folder, "sampled.png"))
            ni = self.num_interpolations
            if num_classes is not None:
                conditional_folder = os.path.join(image_folder, "conditional")
                os.makedirs(conditional_folder, exist_ok=True)
                for i in range(num_classes):
                    with eval_context(m):
                        i_indices = m.sample(batch_size, img_size=img_size, class_idx=i)
                    with eval_context(vqvae):
                        sampled = vqvae.reconstruct_from(i_indices, class_idx=i)
                    sampled_path = os.path.join(conditional_folder, f"sampled_{i}.png")
                    save_images(sampled, sampled_path)
                    with eval_context(m):
                        i1 = m.sample(ni, img_size=img_size, class_idx=i)
                        i2 = m.sample(ni, img_size=img_size, class_idx=i)
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
    vqvae: VQVAE
    tmp_callback_name: Optional[str] = None

    def __init__(
        self,
        config: DLConfig,
        *,
        workspace: str,
        vqvae_log_folder: str,
        num_classes: Optional[int] = None,
        device: device_type = None,
    ):
        self.config = config
        self.device = get_torch_device(device)
        packed_path = os.path.join(vqvae_log_folder, "packed")
        pack(vqvae_log_folder, packed_path)
        self.vqvae = load_inference(packed_path).build_model.model.m
        self.vqvae.to(self.device)
        self.code_export_folder = os.path.join(workspace, "codes")
        if VQVAEInference.tmp_callback_name is not None:
            ImageCallback.remove(VQVAEInference.tmp_callback_name)
        VQVAEInference.tmp_callback_name = register_callback(self.vqvae, num_classes)
        callback_names = config.callback_names or []
        if not isinstance(callback_names, list):
            callback_names = [callback_names]
        callback_names.append(VQVAEInference.tmp_callback_name)
        config.callback_names = callback_names

    def export_code_indices(self, data: IData, export_folder: str) -> None:
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
            batcher = TensorBatcher(loader, self.device)
            for batch in tqdm(batcher, desc=f"{name} codes", total=len(loader)):
                labels.append(batch[LABEL_KEY].cpu())
                net = batch[INPUT_KEY]
                with eval_context(self.vqvae):
                    net = self.vqvae.get_code_indices(net).unsqueeze(1)
                code_indices.append(net.cpu())
                if self.config.is_debug:
                    break
            all_codes = torch.cat(code_indices, dim=0)
            path = os.path.join(export_folder, f"{name}.pt")
            print_info(f"saving {name} codes ({all_codes.shape})")
            torch.save(all_codes, path)
            all_labels = torch.cat(labels, dim=0)
            label_path = os.path.join(export_folder, f"{name}_labels.pt")
            print_info(f"saving {name} labels ({all_labels.shape})")
            torch.save(all_labels, label_path)
        if not self.config.is_debug:
            with open(finished_path, "w"):
                pass

    def fit(self, images: IData, data_config: DataConfig) -> "VQVAEInference":
        export_folder = self.code_export_folder
        self.export_code_indices(images, export_folder)
        x_train = torch.load(os.path.join(export_folder, "train.pt"))
        y_train = torch.load(os.path.join(export_folder, "train_labels.pt"))
        x_valid = torch.load(os.path.join(export_folder, "valid.pt"))
        y_valid = torch.load(os.path.join(export_folder, "valid_labels.pt"))
        tensor_data: ArrayData = ArrayData.init(data_config)
        tensor_data = tensor_data.fit(x_train, y_train, x_valid, y_valid)
        DLTrainingPipeline.init(self.config).fit(tensor_data, device=self.device)
        return self


__all__ = [
    "VQVAEInference",
]
