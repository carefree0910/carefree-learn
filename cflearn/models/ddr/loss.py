import torch

from typing import *
from cftool.misc import LoggingMixin
from torch.nn.functional import l1_loss
from torch.nn.functional import softplus


from ...losses import LossBase
from ...types import tensor_dict_type
from ...modules.auxiliary import MTL


class DDRLoss(LossBase, LoggingMixin):
    def _init_config(self, config: Dict[str, Any]) -> None:
        self.mtl = MTL(18, config["mtl_method"])
        self._lb_recover = config.setdefault("lambda_recover", 1.0)
        self.register_buffer("zero", torch.zeros([1], dtype=torch.float32))

    def _core(  # type: ignore
        self,
        predictions: tensor_dict_type,
        target: torch.Tensor,
        *,
        is_synthetic: bool = False,
    ) -> Tuple[torch.Tensor, tensor_dict_type]:
        # median
        if is_synthetic:
            median_losses = median_recover_losses = None
        else:
            median = predictions["predictions"]
            median_losses = l1_loss(median, target, reduction="none")
            median_inverse = predictions["median_inverse"]
            if median_inverse is None:
                median_recover_losses = self.zero
            else:
                median_recover_losses = torch.abs(median_inverse - 0.5)
                median_recover_losses = self._lb_recover * median_recover_losses
        # quantile losses
        q_batch = predictions["q_batch"]
        assert q_batch is not None
        if is_synthetic:
            quantile_losses = None
        else:
            y = predictions["y"]
            assert y is not None
            quantile_losses = self._quantile_losses(y, target, q_batch)
        # q recover losses
        q_inverse = predictions["q_inverse"]
        if q_inverse is None:
            q_recover_losses = self.zero
        else:
            q_recover_losses = l1_loss(q_inverse, q_batch, reduction="none")
            q_recover_losses = self._lb_recover * q_recover_losses
        # cdf losses
        y_batch = predictions["y_batch"]
        assert y_batch is not None
        if is_synthetic:
            cdf_losses = None
        else:
            cdf_logit = predictions["cdf_logit"]
            assert cdf_logit is not None
            cdf_losses = self._cdf_losses(cdf_logit, target, y_batch)
        # y recover losses
        y_recover_losses = None
        if not is_synthetic:
            y_inverse = predictions["y_inverse"]
            if y_inverse is not None:
                y_recover_losses = l1_loss(y_inverse, y_batch, reduction="none")
                y_recover_losses = self._lb_recover * y_recover_losses
        # pdf losses
        pdf = predictions["pdf"]
        pdf_losses = None if pdf is None else self._pdf_losses(pdf)
        # combine
        losses = {}
        suffix = "" if not is_synthetic else "synthetic_"
        # recover losses
        assert q_recover_losses is not None
        losses[f"{suffix}q_recover"] = q_recover_losses
        if y_recover_losses is not None:
            losses[f"{suffix}y_recover"] = y_recover_losses
        # common
        if not is_synthetic:
            assert median_losses is not None
            assert median_recover_losses is not None
            losses["median"] = median_losses
            losses["median_recover"] = median_recover_losses
            assert quantile_losses is not None
            losses["quantile"] = quantile_losses
            assert cdf_losses is not None
            losses["cdf"] = cdf_losses
        # pdf
        if pdf_losses is not None:
            losses[f"{suffix}pdf"] = pdf_losses
        # mtl
        if not self.mtl.registered:
            self.mtl.register(list(losses.keys()))
        return self.mtl(losses), losses

    def forward(  # type: ignore
        self,
        predictions: tensor_dict_type,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, tensor_dict_type]:
        losses, losses_dict = self._core(predictions, target)
        reduced_losses = self._reduce(losses)
        reduced_losses_dict = {k: self._reduce(v) for k, v in losses_dict.items()}
        return reduced_losses, reduced_losses_dict

    @staticmethod
    def _quantile_losses(
        quantiles: torch.Tensor,
        target: torch.Tensor,
        q_batch: torch.Tensor,
    ) -> torch.Tensor:
        quantile_error = target - quantiles
        q1 = q_batch * quantile_error
        q2 = (q_batch - 1.0) * quantile_error
        return torch.max(q1, q2)

    @staticmethod
    def _cdf_losses(
        cdf_logit: torch.Tensor,
        target: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> torch.Tensor:
        indicative = (target <= y_batch).to(torch.float32)
        return -indicative * cdf_logit + softplus(cdf_logit)

    @staticmethod
    def _pdf_losses(pdf: torch.Tensor) -> torch.Tensor:
        negative_mask = pdf <= 1e-8
        losses = torch.zeros_like(pdf)
        losses[negative_mask] = -pdf[negative_mask]
        return losses


__all__ = ["DDRLoss"]
