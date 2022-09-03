import os
import torch

import torch.nn.functional as F

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Optional
from typing import NamedTuple
from cftool.misc import print_warning
from cftool.types import tensor_dict_type

from ..protocol import ITrainer
from ..protocol import ILoss
from ..protocol import IDLModel
from ..constants import LOSS_KEY
from ..constants import LABEL_KEY
from ..constants import LATENT_KEY
from ..constants import PREDICTIONS_KEY
from ..misc.toolkit import set_requires_grad


class ICustomLossOutput(NamedTuple):
    forward_results: tensor_dict_type
    loss_dict: tensor_dict_type


class ICustomLossModule(torch.nn.Module):
    @abstractmethod
    def get_losses(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: ITrainer,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> ICustomLossOutput:
        pass

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass


class IBAKE(ICustomLossModule, metaclass=ABCMeta):
    lb: float
    bake_loss: ILoss
    w_ensemble: float
    is_classification: bool

    def _init_bake(
        self,
        lb: float,
        bake_loss: str,
        bake_loss_config: Optional[Dict[str, Any]],
        w_ensemble: float,
        is_classification: bool,
    ) -> None:
        self.lb = lb
        self.w_ensemble = w_ensemble
        self.is_classification = is_classification
        if bake_loss == "auto":
            bake_loss = "focal" if is_classification else "mae"
        self.bake_loss = ILoss.make(bake_loss, bake_loss_config or {})

    def get_losses(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: ITrainer,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> ICustomLossOutput:
        state = trainer.state
        forward_results = self(batch_idx, batch, state, **forward_kwargs)
        loss_dict = trainer.loss(forward_results, batch, state, **loss_kwargs)
        loss = loss_dict[LOSS_KEY]
        predictions = forward_results[PREDICTIONS_KEY]
        # BAKE
        latent = F.normalize(forward_results[LATENT_KEY])
        batch_size = latent.shape[0]
        eye = torch.eye(batch_size, device=latent.device)
        similarities = F.softmax(latent.mm(latent.t()) - eye * 1.0e9, dim=1)
        inv = (eye - self.w_ensemble * similarities).inverse()
        weights = (1.0 - self.w_ensemble) * inv
        if not self.is_classification:
            bake_inp = predictions
            soft_labels = weights.mm(predictions).detach()
        else:
            bake_inp = F.log_softmax(predictions)
            soft_labels = weights.mm(F.softmax(predictions, dim=1)).detach()
        bake_loss = self.bake_loss(
            {PREDICTIONS_KEY: bake_inp},
            {LABEL_KEY: soft_labels},
            state,
            **loss_kwargs,
        )[LOSS_KEY]
        loss_dict["bake"] = bake_loss
        loss_dict[LOSS_KEY] = loss + self.lb * bake_loss
        return ICustomLossOutput(forward_results, loss_dict)


class IRDropout(ICustomLossModule, metaclass=ABCMeta):
    lb: float
    is_classification: bool

    def get_losses(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: ITrainer,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> ICustomLossOutput:
        state = trainer.state
        fr1 = self(batch_idx, batch, state, **forward_kwargs)
        fr2 = self(batch_idx, batch, state, **forward_kwargs)
        loss_dict = trainer.loss(fr1, batch, state, **loss_kwargs)
        for k, v in trainer.loss(fr2, batch, state, **loss_kwargs).items():
            loss_dict[k] = loss_dict[k] + v
        loss = loss_dict[LOSS_KEY]
        # R-Dropout loss
        p1 = fr1[PREDICTIONS_KEY]
        p2 = fr2[PREDICTIONS_KEY]
        if not self.is_classification:
            r_dropout = ((p1 - p2) ** 2).mean()
        else:
            p1_loss = F.kl_div(F.log_softmax(p1, dim=-1), F.softmax(p2, dim=-1))
            p2_loss = F.kl_div(F.log_softmax(p2, dim=-1), F.softmax(p1, dim=-1))
            r_dropout = 0.5 * (p1_loss + p2_loss)
        loss_dict["r_dropout"] = r_dropout
        loss_dict[LOSS_KEY] = loss + self.lb * r_dropout
        return ICustomLossOutput(fr1, loss_dict)


class CascadeMixin:
    lv1_net: IDLModel
    lv2_net: IDLModel

    def _construct(
        self,
        lv1_model_name: str,
        lv2_model_name: str,
        lv1_model_config: Dict[str, Any],
        lv2_model_config: Dict[str, Any],
        lv1_model_ckpt_path: Optional[str],
        lv1_model_trainable: bool,
    ) -> None:
        self.lv1_net = IDLModel.make(lv1_model_name, lv1_model_config)
        if lv1_model_ckpt_path is not None:
            if not os.path.isfile(lv1_model_ckpt_path):
                print_warning(f"'{lv1_model_ckpt_path}' does not exist")
            else:
                state_dict = torch.load(lv1_model_ckpt_path, map_location="cpu")
                self.lv1_net.load_state_dict(state_dict)
        if not lv1_model_trainable:
            set_requires_grad(self.lv1_net, False)
        self.lv2_net = IDLModel.make(lv2_model_name, lv2_model_config)


__all__ = [
    "ICustomLossModule",
    "IBAKE",
    "IRDropout",
    "CascadeMixin",
]
