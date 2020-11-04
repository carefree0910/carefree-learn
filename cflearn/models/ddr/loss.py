import torch

from typing import *
from cftool.misc import LoggingMixin
from torch.nn.functional import l1_loss
from torch.nn.functional import mse_loss
from torch.nn.functional import softplus

from ...losses import LossBase
from ...types import tensor_dict_type
from ...modules.auxiliary import MTL


class DDRLoss(LossBase, LoggingMixin):
    def _init_config(self, config: Dict[str, Any]) -> None:
        self.mtl = MTL(20, config["mtl_method"])
        self._lb_pdf = config.setdefault("lambda_pdf", 0.01)
        self._pdf_eps = config.setdefault("pdf_eps", 1.0e-8)
        self._lb_recover = config.setdefault("lambda_recover", 1.0)
        self._lb_latent = config.setdefault("lambda_latent", 10.0)

    def _core(  # type: ignore
        self,
        predictions: tensor_dict_type,
        target: torch.Tensor,
        *,
        is_synthetic: bool = False,
    ) -> Tuple[torch.Tensor, tensor_dict_type]:
        # median
        if is_synthetic:
            median_losses = median_ae_losses = median_recover_losses = None
        else:
            median = predictions["predictions"]
            median_losses = l1_loss(median, target, reduction="none")
            median_ae = predictions["median_ae"]
            median_ae_losses = torch.abs(median_ae - 0.5)
            median_ae_losses = self._lb_recover * median_ae_losses
            median_inverse = predictions["median_inverse"]
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
        # q auto encode losses
        q_ae = predictions["q_ae"]
        if q_ae is None:
            q_ae_losses = None
        else:
            q_ae_losses = l1_loss(q_ae, q_batch, reduction="none")
            q_ae_losses = self._lb_recover * q_ae_losses
        # q recover losses
        q_inverse = predictions["q_inverse"]
        q_recover_losses = l1_loss(q_inverse, q_batch, reduction="none")
        q_recover_losses = self._lb_recover * q_recover_losses
        # q latent losses
        q_latent = predictions["q_latent"]
        qy_latent = predictions["qy_latent"]
        y_inverse_latent = predictions["y_inverse_latent"]
        yq_inverse_latent = predictions["yq_inverse_latent"]
        q_latent_losses1 = mse_loss(qy_latent, y_inverse_latent.detach())
        q_latent_losses2 = mse_loss(yq_inverse_latent, q_latent.detach())
        q_latent_losses = self._lb_latent * (q_latent_losses1 + q_latent_losses2)
        # cdf losses
        y_batch = predictions["y_batch"]
        assert y_batch is not None
        if is_synthetic:
            cdf_losses = None
        else:
            cdf_logit = predictions["cdf_logit"]
            assert cdf_logit is not None
            cdf_losses = self._cdf_losses(cdf_logit, target, y_batch)
        # y auto encode & recover losses
        y_ae_losses = y_recover_losses = None
        if not is_synthetic:
            y_ae = predictions["y_ae"]
            if y_ae is not None:
                y_ae_losses = l1_loss(y_ae, y_batch, reduction="none")
                y_ae_losses = self._lb_recover * y_ae_losses
            y_inverse = predictions["y_inverse"]
            if y_inverse is not None:
                y_recover_losses = l1_loss(y_inverse, y_batch, reduction="none")
                y_recover_losses = self._lb_recover * y_recover_losses
        # y latent losses
        y_latent = predictions["y_latent"]
        yq_latent = predictions["yq_latent"]
        q_inverse_latent = predictions["q_inverse_latent"]
        y_latent_losses1 = mse_loss(yq_latent, q_inverse_latent.detach())
        qy_inverse_latent = predictions["qy_inverse_latent"]
        if qy_inverse_latent is None:
            y_latent_losses = y_latent_losses1
        else:
            y_latent_losses2 = mse_loss(qy_inverse_latent, y_latent.detach())
            y_latent_losses = y_latent_losses1 + y_latent_losses2
        y_latent_losses = self._lb_latent * y_latent_losses
        # pdf losses
        pdf = predictions["pdf"]
        pdf_losses = None if pdf is None else self._pdf_losses(pdf, is_synthetic)
        # combine
        losses = {}
        suffix = "" if not is_synthetic else "synthetic_"
        # recover losses
        assert q_recover_losses is not None
        losses[f"{suffix}q_recover"] = q_recover_losses
        if y_recover_losses is not None:
            losses[f"{suffix}y_recover"] = y_recover_losses
        # latent losses
        assert q_latent_losses is not None
        assert y_latent_losses is not None
        losses[f"{suffix}q_latent"] = q_latent_losses
        losses[f"{suffix}y_latent"] = y_latent_losses
        # common
        if not is_synthetic:
            assert median_ae_losses is not None
            assert q_ae_losses is not None
            assert y_ae_losses is not None
            losses["median_ae"] = median_ae_losses
            losses["q_ae"] = q_ae_losses
            losses["y_ae"] = y_ae_losses
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

    def _pdf_losses(self, pdf: torch.Tensor, is_synthetic: bool) -> torch.Tensor:
        negative_mask = pdf <= self._pdf_eps
        losses = torch.zeros_like(pdf)
        losses[negative_mask] = -pdf[negative_mask]
        if not is_synthetic:
            positive_mask = ~negative_mask
            losses[positive_mask] = -self._lb_pdf * torch.log(pdf[positive_mask])
        return losses


__all__ = ["DDRLoss"]
