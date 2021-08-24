import torch

import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from torch.nn.parallel import DistributedDataParallel as DDP

from ..encoder import Encoder1DBase
from ....types import tensor_dict_type
from ....protocol import StepOutputs
from ....protocol import TrainerState
from ....protocol import MetricsOutputs
from ....protocol import ModelWithCustomSteps
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....misc.toolkit import to_device
from ....misc.toolkit import has_batch_norms
from ....misc.internal_ import CVLoader


def world_size() -> int:
    return 1 if not dist.is_initialized() else dist.get_world_size()


def _get_dino_defaults(name: str) -> Dict[str, Any]:
    if name == "vit":
        return {"patch_size": 16, "drop_path_rate": 0.1}
    return {}


class Scheduler:
    def __init__(self, values: np.ndarray):
        self.values = values
        self.max_idx = len(values) - 1

    def __getitem__(self, index: int) -> Any:
        return self.values[min(index, self.max_idx)]


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    num_step_per_epoch: int,
    warmup_epochs: int = 0,
    start_warmup_value: int = 0,
) -> Scheduler:
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * num_step_per_epoch
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * num_step_per_epoch - warmup_iters)
    diff = base_value - final_value
    schedule = final_value + 0.5 * diff * (1.0 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * num_step_per_epoch
    return Scheduler(schedule)


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        *,
        img_end_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> Tensor:
        img_crops = batch[INPUT_KEY]
        if not isinstance(img_crops, list):
            img_crops = batch[INPUT_KEY] = [img_crops]
        if img_end_idx is not None:
            img_crops = img_crops[:img_end_idx]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([img_crop.shape[-1] for img_crop in img_crops]),
                return_counts=True,
            )[1],
            0,
        )
        outputs = []
        start_idx = 0
        for end_idx in idx_crops:
            local_batch = shallow_copy_dict(batch)
            local_batch[INPUT_KEY] = torch.cat(img_crops[start_idx:end_idx])
            idx_rs = self.backbone(batch_idx, local_batch, state, **kwargs)
            idx_out = idx_rs[LATENT_KEY]
            if isinstance(idx_out, tuple):
                idx_out = idx_out[0]
            outputs.append(idx_out)
            start_idx = end_idx
        return self.head(torch.cat(outputs))


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        batch_norm: bool = False,
        norm_last_layer: bool = True,
        *,
        num_layers: int = 3,
        latent_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.mapping = nn.Linear(in_dim, bottleneck_dim)
        else:
            blocks = [nn.Linear(in_dim, latent_dim)]
            if batch_norm:
                blocks.append(nn.BatchNorm1d(latent_dim))
            blocks.append(nn.GELU())
            for _ in range(num_layers - 2):
                blocks.append(nn.Linear(latent_dim, latent_dim))
                if batch_norm:
                    blocks.append(nn.BatchNorm1d(latent_dim))
                blocks.append(nn.GELU())
            blocks.append(nn.Linear(latent_dim, bottleneck_dim))
            self.mapping = nn.Sequential(*blocks)
        self.apply(self._init_weights)
        last = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(last)
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, net: Tensor) -> Tensor:
        net = self.mapping(net)
        net = nn.functional.normalize(net, dim=-1, p=2)
        net = self.last_layer(net)
        return net


class DINOLoss(nn.Module):
    center: torch.Tensor

    def __init__(
        self,
        out_dim: int,
        teacher_temp: float,
        warmup_teacher_temp: float,
        warmup_teacher_temp_epochs: int,
        teacher_temp_epochs: int,
        *,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        teacher_temp_constant_epochs = teacher_temp_epochs - warmup_teacher_temp_epochs
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp,
                    teacher_temp,
                    warmup_teacher_temp_epochs,
                ),
                np.ones(teacher_temp_constant_epochs) * teacher_temp,
            )
        )
        self.num_epochs = teacher_temp_epochs

    def forward(
        self,
        epoch: int,
        num_crops: int,
        student_output: Tensor,
        teacher_output: Tensor,
    ) -> Tensor:
        student_logits = student_output / self.student_temp
        student_logits_list = student_logits.chunk(num_crops)

        temp = self.teacher_temp_schedule[min(epoch, self.num_epochs)]
        teacher_logits = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_logits_list = teacher_logits.detach().chunk(2)

        total_loss = 0.0
        n_loss_terms = 0
        for it, t_logit in enumerate(teacher_logits_list):
            for iv, v_logit in enumerate(student_logits_list):
                if iv == it:
                    continue
                loss = torch.sum(-t_logit * F.log_softmax(v_logit, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: Tensor) -> None:
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * world_size())
        m = self.center_momentum
        self.center = self.center * m + batch_center * (1.0 - m)


@ModelWithCustomSteps.register("dino")
class DINO(ModelWithCustomSteps):
    custom_params_groups = True
    custom_ddp_initialization = True

    lr_schedule: Optional[Scheduler]
    wd_schedule: Optional[Scheduler]
    momentum_schedule: Optional[Scheduler]

    def __init__(
        self,
        encoder_name: str = "vit",
        encoder_configs: Optional[Dict[str, Any]] = None,
        student_specific: Optional[Dict[str, Any]] = None,
        teacher_specific: Optional[Dict[str, Any]] = None,
        *,
        out_dim: int = 65536,
        use_bn_in_head: bool = False,
        norm_last_layer: bool = True,
        teacher_temp: float = 0.07,
        momentum_teacher: float = 0.996,
        warmup_teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        teacher_temp_epochs: int,
        freeze_last_layer: int = 1,
        weight_decay: float = 0.04,
        weight_decay_end: float = 0.4,
        warmup_epochs: int = 10,
    ):
        super().__init__()
        base = update_dict(encoder_configs or {}, _get_dino_defaults(encoder_name))
        student_cfg = update_dict(student_specific or {}, shallow_copy_dict(base))
        teacher_cfg = update_dict(teacher_specific or {}, shallow_copy_dict(base))
        student = Encoder1DBase.make(encoder_name, student_cfg)
        teacher = Encoder1DBase.make(encoder_name, teacher_cfg)
        self.ddp_student = self.ddp_teacher = None
        self.student = MultiCropWrapper(
            student,
            DINOHead(
                student.latent_dim,
                out_dim,
                use_bn_in_head,
                norm_last_layer,
            ),
        )
        self.teacher = MultiCropWrapper(
            teacher,
            DINOHead(teacher.latent_dim, out_dim, use_bn_in_head),
        )
        self.freeze_last_layer = freeze_last_layer
        self.teacher.load_state_dict(self.student.state_dict())
        self.loss = DINOLoss(
            out_dim,
            teacher_temp,
            warmup_teacher_temp,
            warmup_teacher_temp_epochs,
            teacher_temp_epochs,
        )
        self.momentum_teacher = momentum_teacher
        self.teacher_temp_epochs = teacher_temp_epochs
        self.weight_decay = weight_decay
        self.weight_decay_end = weight_decay_end
        self.warmup_epochs = warmup_epochs
        self.lr_schedule = None
        self.wd_schedule = None
        self.momentum_schedule = None

    @property
    def student_for_training(self) -> MultiCropWrapper:
        return self.ddp_student or self.student

    @property
    def teacher_for_training(self) -> MultiCropWrapper:
        return self.ddp_teacher or self.teacher

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return self.student.backbone(batch_idx, batch, state, **kwargs)

    def summary_forward(self, batch_idx: int, batch: tensor_dict_type) -> None:
        self.student(batch_idx, to_device(batch, self.device))

    def _get_loss(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: Any,
        forward_kwargs: Dict[str, Any],
    ) -> Tuple[tensor_dict_type, Tensor]:
        state = trainer.state
        with torch.cuda.amp.autocast(enabled=trainer.use_amp):
            teacher_output = self.teacher_for_training(
                batch_idx,
                batch,
                state,
                img_end_idx=2,
                **forward_kwargs,
            )
            student_output = self.student_for_training(
                batch_idx,
                batch,
                state,
                **forward_kwargs,
            )
            epoch = state.epoch
            num_crops = len(batch[INPUT_KEY])
            loss = self.loss(epoch, num_crops, student_output, teacher_output)
        return {"student": student_output, "teacher": teacher_output}, loss

    def train_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: Any,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        state = trainer.state
        if self.lr_schedule is None:
            self.lr_schedule = cosine_scheduler(
                self.lr * (len(batch[INPUT_KEY]) * world_size()) / 256.0,  # type: ignore
                self.min_lr,
                self.teacher_temp_epochs,
                state.num_step_per_epoch,
                warmup_epochs=self.warmup_epochs,
            )
        if self.wd_schedule is None:
            self.wd_schedule = cosine_scheduler(
                self.weight_decay,
                self.weight_decay_end,
                self.teacher_temp_epochs,
                state.num_step_per_epoch,
            )
        # manual scheduling
        optimizer = trainer.optimizers["all"]
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[state.step]
            if i == 0:
                param_group["weight_decay"] = self.wd_schedule[state.step]
        # forward pass
        rs, loss = self._get_loss(batch_idx, batch, trainer, forward_kwargs)
        # backward pass
        optimizer.zero_grad()
        trainer.grad_scaler.scale(loss).backward()
        # clip norm
        if trainer.clip_norm > 0.0:
            trainer.grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                self.student_for_training.parameters(),
                max_norm=trainer.clip_norm,
            )
        # freeze last layer
        if state.epoch <= self.freeze_last_layer:
            for n, p in self.student.named_parameters():
                if "last_layer" in n:
                    p.grad = None
        # update parameters
        trainer.grad_scaler.step(optimizer)
        trainer.grad_scaler.update()
        # update momentum teacher
        if self.momentum_schedule is None:
            self.momentum_schedule = cosine_scheduler(
                self.momentum_teacher,
                1.0,
                self.teacher_temp_epochs,
                state.num_step_per_epoch,
            )
        with torch.no_grad():
            m = self.momentum_schedule[state.step]
            for param_q, param_k in zip(
                self.student.parameters(),
                self.teacher.parameters(),
            ):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)
        # return
        return StepOutputs(rs, {LOSS_KEY: loss})

    def evaluate_step(  # type: ignore
        self,
        loader: CVLoader,
        portion: float,
        trainer: Any,
    ) -> MetricsOutputs:
        losses = []
        for i, batch in enumerate(loader):
            if i / len(loader) >= portion:
                break
            batch = to_device(batch, self.device)
            losses.append(self._get_loss(i, batch, trainer, {})[1].item())
        # gather
        mean_loss = sum(losses) / len(losses)
        return MetricsOutputs(
            -mean_loss,
            {
                "loss": mean_loss,
                "lr": self.lr_schedule[trainer.state.step],  # type: ignore
                "wd": self.wd_schedule[trainer.state.step],  # type: ignore
            },
        )

    @staticmethod
    def params_groups(m: nn.Module) -> Any:
        regularized = []
        bias_and_norm = []
        for name, param in m.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or len(param.shape) == 1:
                bias_and_norm.append(param)
            else:
                regularized.append(param)
        return [{"params": regularized}, {"params": bias_and_norm, "weight_decay": 0.0}]

    def _init_with_trainer(self, trainer: Any) -> None:
        self.teacher_for_training.requires_grad_(False)

    def init_ddp(self, trainer: Any) -> None:
        if has_batch_norms(self.student):
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
            self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)
        self.ddp_student = DDP(self.student, device_ids=[trainer.rank])
        self.ddp_teacher = DDP(self.teacher, device_ids=[trainer.rank])
        self.ddp_teacher.requires_grad_(False)  # type: ignore

    def permute_trainer_config(self, trainer_config: Dict[str, Any]) -> None:
        # TODO : make `permute_trainer_config` more general
        if trainer_config["clip_norm"] == 0.0:
            trainer_config["clip_norm"] = 3.0
        if trainer_config["lr"] is None:
            trainer_config["lr"] = 0.0005
        self.lr = trainer_config["lr"]
        self.min_lr = trainer_config.pop("min_lr", 1.0e-6)
        if trainer_config["optimizer_name"] is None:
            trainer_config["optimizer_name"] = "adamw"
        trainer_config["scheduler_name"] = "none"


__all__ = [
    "DINO",
]