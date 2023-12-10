import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any, Tuple
from typing import Dict
from typing import List
from typing import Optional
from cftool.types import tensor_dict_type

from ...schema import IMetric
from ...schema import DLConfig
from ...schema import IDLModel
from ...schema import ITrainer
from ...schema import TrainStep
from ...schema import IInference
from ...schema import IDataLoader
from ...schema import TrainerState
from ...schema import TrainerConfig
from ...schema import TrainStepLoss
from ...schema import MetricsOutputs
from ...schema import InferenceOutputs
from ...modules import build_generator
from ...modules import DDPM
from ...constants import INPUT_KEY
from ...constants import PREDICTIONS_KEY
from ...modules.multimodal.diffusion.utils import extract_to


class DDPMStep(TrainStep):
    def setup(
        self,
        loss_type: str = "l2",
        l_simple_weight: float = 1.0,
        original_elbo_weight: float = 0.0,
    ) -> None:
        self.loss_type = loss_type
        self.l_simple_weight = l_simple_weight
        self.original_elbo_weight = original_elbo_weight

    def loss_fn(
        self,
        m: "DDPMModel",
        state: Optional[TrainerState],
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> TrainStepLoss:
        ddpm = m.m
        noise = forward_results[ddpm.noise_key]
        unet_out = forward_results[PREDICTIONS_KEY]
        timesteps = forward_results[ddpm.timesteps_key]
        if ddpm.parameterization == "eps":
            target = noise
        elif ddpm.parameterization == "x0":
            target = batch[INPUT_KEY]
        elif ddpm.parameterization == "v":
            target = self.get_v(ddpm, batch[INPUT_KEY], noise, timesteps)
        else:
            msg = f"unrecognized parameterization '{ddpm.parameterization}' occurred"
            raise ValueError(msg)

        losses = {}
        if self.loss_type == "l1":
            loss = (unet_out - target).abs()
        elif self.loss_type == "l2":
            loss = F.mse_loss(unet_out, target, reduction="none")
        else:
            raise ValueError(f"unrecognized loss '{self.loss_type}' occurred")
        loss = loss.mean(dim=(1, 2, 3))
        loss_simple = loss
        losses["simple"] = loss_simple.mean().item()

        log_var_t = ddpm.log_var.to(unet_out.device)[timesteps]
        loss_simple = loss_simple / torch.exp(log_var_t) + log_var_t
        if ddpm.learn_log_var:
            losses["gamma"] = loss_simple.mean().item()
            losses["log_var"] = ddpm.log_var.data.mean().item()

        loss_simple = self.l_simple_weight * loss_simple.mean()
        if self.original_elbo_weight <= 0:
            losses["loss"] = loss_simple.item()
            return TrainStepLoss(loss_simple, losses)

        loss_vlb = (ddpm.lvlb_weights[timesteps] * loss).mean()
        losses["vlb"] = loss_vlb.item()

        loss_vlb = self.original_elbo_weight * loss_vlb
        loss = loss_simple + loss_vlb
        losses["loss"] = loss.item()
        return TrainStepLoss(loss, losses)

    def get_v(self, ddpm: DDPM, x: Tensor, noise: Tensor, ts: Tensor) -> Tensor:
        num_dim = x.ndim
        return (
            extract_to(ddpm.sqrt_alphas_cumprod, ts, num_dim) * noise
            - extract_to(ddpm.sqrt_one_minus_alphas_cumprod, ts, num_dim) * x
        )

    def callback(
        self,
        m: "DDPMModel",
        trainer: ITrainer,
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
    ) -> None:
        ddpm = m.m
        if ddpm.training and ddpm.unet_ema is not None:
            ddpm.unet_ema()


@IDLModel.register("ddpm")
class DDPMModel(IDLModel):
    m: DDPM

    @property
    def train_steps(self) -> List[TrainStep]:
        loss_config = self.config.loss_config or {}
        step = DDPMStep("learnable")
        step.setup(**loss_config)
        return [step]

    @property
    def all_modules(self) -> List[nn.Module]:
        return [self.m]

    @property
    def learnable(self) -> List[nn.Parameter]:
        ddpm = self.m
        params = list(ddpm.unet.parameters())
        if ddpm.learn_log_var:
            params.append(ddpm.log_var)
        return params

    def build(self, config: DLConfig) -> None:
        self.m = build_generator(config.module_name, config=config.module_config)

    def evaluate(
        self,
        config: TrainerConfig,
        metrics: Optional[IMetric],
        inference: IInference,
        loader: IDataLoader,
        *,
        portion: float = 1,
        state: Optional[TrainerState] = None,
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> MetricsOutputs:
        def get_outputs() -> InferenceOutputs:
            return inference.get_outputs(
                loader,
                portion=portion,
                use_losses_as_metrics=True,
                return_outputs=False,
                **(forward_kwargs or {}),
            )

        # TODO : specify timesteps & noise to make results deterministic
        outputs = get_outputs()
        losses = outputs.loss_items
        if losses is None:
            raise ValueError("`loss_items` should not be None")
        score = -losses["simple"]
        # no ema
        ddpm = self.m
        if ddpm.unet_ema is None:
            return MetricsOutputs(score, losses, {k: False for k in losses})
        losses = {f"{k}_ema": v for k, v in losses.items()}
        ddpm.unet_ema.train()
        outputs = get_outputs()
        if outputs.loss_items is None:
            raise ValueError("`loss_items` should not be None")
        losses.update(outputs.loss_items)
        ddpm.unet_ema.eval()
        return MetricsOutputs(score, losses, {k: False for k in losses})

    def get_forward_args(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> Tuple[Any, ...]:
        return batch[INPUT_KEY], batch.get(self.m.cond_key)


__all__ = [
    "DDPMModel",
]
