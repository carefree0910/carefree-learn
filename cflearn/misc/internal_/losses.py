import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from cftool.misc import shallow_copy_dict

from ...types import losses_type
from ...types import tensor_dict_type
from ...protocol import LossProtocol
from ...protocol import TrainerState
from ...constants import LOSS_KEY
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY
from ...constants import PREDICTIONS_KEY
from ...misc.toolkit import iou
from ...misc.toolkit import corr
from ...misc.toolkit import to_torch


@LossProtocol.register("iou")
class IOULoss(LossProtocol):
    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        logits = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return 1.0 - iou(logits, labels)


@LossProtocol.register("bce")
class BCELoss(LossProtocol):
    def _init_config(self) -> None:
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        losses = self.bce(predictions, labels)
        return losses.mean(tuple(range(1, len(losses.shape))))


@LossProtocol.register("mae")
class MAELoss(LossProtocol):
    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return F.l1_loss(predictions, labels, reduction="none")


@LossProtocol.register("sigmoid_mae")
class SigmoidMAELoss(LossProtocol):
    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        losses = F.l1_loss(torch.sigmoid(predictions), labels, reduction="none")
        return losses.mean((1, 2, 3))


@LossProtocol.register("mse")
class MSELoss(LossProtocol):
    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return F.mse_loss(predictions, labels, reduction="none")


@LossProtocol.register("recon")
class ReconstructionLoss(LossProtocol):
    def _init_config(self) -> None:
        base_loss_name = self.config.pop("base_loss", None)
        if base_loss_name is None:
            raise ValueError("`base_loss` should be provided for `ReconstructionLoss`")
        self.base_loss = LossProtocol.make(base_loss_name, self.config)

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        batch = shallow_copy_dict(batch)
        batch[LABEL_KEY] = batch[INPUT_KEY]
        return self.base_loss._core(forward_results, batch, state, **kwargs)


@LossProtocol.register("quantile")
class QuantileLoss(LossProtocol):
    def _init_config(self) -> None:
        q = self.config.get("q")
        if q is None:
            raise ValueError("'q' should be provided in Quantile loss")
        if isinstance(q, float):
            self.register_buffer("q", torch.tensor([q], torch.float32))
        else:
            q = np.asarray(q, np.float32).reshape([1, -1])
            self.register_buffer("q", to_torch(q))

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        quantile_error = batch[LABEL_KEY] - forward_results[PREDICTIONS_KEY]
        neg_errors = self.q * quantile_error  # type: ignore
        pos_errors = (self.q - 1.0) * quantile_error  # type: ignore
        quantile_losses = torch.max(neg_errors, pos_errors)
        return quantile_losses.mean(1, keepdim=True)


@LossProtocol.register("corr")
class CorrelationLoss(LossProtocol):
    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return -corr(predictions, labels, get_diagonal=True)


@LossProtocol.register("cross_entropy")
class CrossEntropyLoss(LossProtocol):
    @staticmethod
    def _get_stat(
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_prob_mat = F.log_softmax(predictions, dim=1)
        nll_losses = -log_prob_mat.gather(dim=1, index=labels)
        return log_prob_mat, nll_losses

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return self._get_stat(predictions, labels)[1]


@LossProtocol.register("label_smooth_cross_entropy")
class LabelSmoothCrossEntropyLoss(LossProtocol):
    def _init_config(self) -> None:
        self._eps = self.config.setdefault("eps", 0.1)

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        log_prob_mat, nll_losses = CrossEntropyLoss._get_stat(predictions, labels)
        smooth_losses = -log_prob_mat.sum(dim=1, keepdim=True)
        eps = self._eps / log_prob_mat.shape[-1]
        return (1.0 - self._eps) * nll_losses + eps * smooth_losses


@LossProtocol.register("focal")
class FocalLoss(LossProtocol):
    def _init_config(self) -> None:
        self._input_logits = self.config.setdefault("input_logits", True)
        self._eps = self.config.setdefault("eps", 1e-6)
        self._gamma = self.config.setdefault("gamma", 2.0)
        alpha = self.config.setdefault("alpha", None)
        if isinstance(alpha, (int, float)):
            alpha = [alpha, 1 - alpha]
        elif isinstance(alpha, (list, tuple)):
            alpha = list(alpha)
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", to_torch(np.array(alpha, np.float32)))

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        if not self._input_logits:
            prob_mat = predictions.view(-1, predictions.shape[-1]) + self._eps
        else:
            logits_mat = predictions.view(-1, predictions.shape[-1])
            prob_mat = F.softmax(logits_mat, dim=1) + self._eps
        gathered_prob_flat = prob_mat.gather(dim=1, index=labels).view(-1)
        gathered_log_prob_flat = gathered_prob_flat.log()
        if self.alpha is not None:
            alpha_target = self.alpha.gather(dim=0, index=labels.view(-1))
            gathered_log_prob_flat = gathered_log_prob_flat * alpha_target
        return -gathered_log_prob_flat * (1 - gathered_prob_flat) ** self._gamma


multi_prefix_mapping: Dict[str, Type["MultiLoss"]] = {}


class MultiLoss(LossProtocol, metaclass=ABCMeta):
    prefix: str

    names: Union[str, List[str]]
    base_losses: nn.ModuleList

    def _init_config(self) -> None:
        if isinstance(self.names, str):
            base_losses = [LossProtocol.make(self.names, self.config)]
        else:
            base_losses = [
                LossProtocol.make(name, self.config.get(name, {}))
                for name in self.names
            ]
        self.base_losses = nn.ModuleList(base_losses)

    @staticmethod
    def _inject(key: str, base_losses: losses_type, all_losses: losses_type) -> None:
        if isinstance(base_losses, dict):
            base_losses = base_losses[LOSS_KEY]
        all_losses[key] = base_losses

    @classmethod
    def register_(
        cls,
        base_loss_names: Union[str, List[str]],
        *,
        tag: Optional[str] = None,
    ) -> str:
        if tag is None:
            if isinstance(base_loss_names, str):
                tag = f"{cls.prefix}_{base_loss_names}"
            else:
                tag = f"{cls.prefix}_{'_'.join(base_loss_names)}"
        if tag in cls.d:
            return tag

        @cls.register(tag)
        class _(cls):  # type: ignore
            names = base_loss_names

        return tag

    @classmethod
    def record_prefix(cls) -> Callable[[Type["MultiLoss"]], Type["MultiLoss"]]:
        def _(cls_: Type[MultiLoss]) -> Type[MultiLoss]:
            global multi_prefix_mapping
            multi_prefix_mapping[cls_.prefix] = cls_
            return cls_

        return _


@MultiLoss.record_prefix()
class MultiTaskLoss(MultiLoss):
    prefix = "multi_task"

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        losses: losses_type = {}
        for loss_ins in self.base_losses:
            self._inject(
                loss_ins.__identifier__,
                loss_ins._core(forward_results, batch, state, **kwargs),
                losses,
            )
        losses[LOSS_KEY] = sum(losses.values())
        return losses


@MultiLoss.record_prefix()
class MultiStageLoss(MultiLoss):
    prefix = "multi_stage"

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        losses: losses_type = {}
        for i, pred in enumerate(predictions):
            forward_results[PREDICTIONS_KEY] = pred
            for loss_ins in self.base_losses:
                self._inject(
                    f"{loss_ins.__identifier__}{i}",
                    loss_ins._core(forward_results, batch, state, **kwargs),
                    losses,
                )
        losses[LOSS_KEY] = sum(losses.values())
        return losses


__all__ = [
    "IOULoss",
    "MAELoss",
    "MSELoss",
    "QuantileLoss",
    "SigmoidMAELoss",
    "CorrelationLoss",
    "CrossEntropyLoss",
    "LabelSmoothCrossEntropyLoss",
    "FocalLoss",
    "MultiStageLoss",
]
