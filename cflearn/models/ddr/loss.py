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
        self.fetch_q = config["fetch_q"]
        self.fetch_cdf = config["fetch_cdf"]
        self.mtl = MTL(18, config["mtl_method"])
        self._lb_pdf = config.setdefault("lambda_pdf", 0.01)
        self._pdf_eps = config.setdefault("pdf_eps", 1.0e-8)
        self._lb_recover = config.setdefault("lambda_recover", 1.0)

    def _q_losses(
        self,
        predictions: tensor_dict_type,
        target: torch.Tensor,
        is_synthetic: bool,
    ) -> tensor_dict_type:
        # median affine
        if is_synthetic:
            syn_med_add = predictions["syn_med_add"].abs()
            syn_med_mul = predictions["syn_med_mul"].abs()
            syn_med_res = predictions["syn_med_res"].detach()
            mr_anchor_losses = (syn_med_res * (1.0 - syn_med_mul) - syn_med_add).abs()
            return {"median_residual_anchor": mr_anchor_losses}
        # median
        median = predictions["predictions"]
        median_losses = l1_loss(median, target, reduction="none")
        # median residual
        q_sign = predictions["q_sign"]
        median_residual = predictions["med_res"]
        target_residual = target - median.detach()
        tmr_sign = torch.sign(target_residual)
        same_sign_mask = q_sign * tmr_sign > 0
        tmr = (target_residual * tmr_sign)[same_sign_mask]
        mr = median_residual[same_sign_mask]
        mr_losses = torch.zeros_like(target_residual)
        mr_losses[same_sign_mask] = torch.abs(tmr - mr)
        # quantile
        q_batch = predictions["q_batch"]
        assert q_batch is not None
        y_res = predictions["y_res"]
        assert y_res is not None
        quantile_losses = self._quantile_losses(y_res, target_residual, q_batch)
        # combine
        return {
            "median": median_losses,
            "quantile": quantile_losses,
            "median_residual": mr_losses,
        }

    def _y_losses(
        self,
        predictions: tensor_dict_type,
        target: torch.Tensor,
        is_synthetic: bool,
    ) -> tensor_dict_type:
        # cdf
        cdf_losses = None
        if not is_synthetic:
            y_batch = predictions["y_batch"]
            cdf_logit = predictions["cdf_logit"]
            assert y_batch is not None and cdf_logit is not None
            cdf_losses = self._cdf_losses(cdf_logit, target, y_batch)
        # pdf
        pdf = predictions["pdf"]
        pdf_losses = None if pdf is None else self._pdf_losses(pdf, is_synthetic)
        # combine
        losses = {}
        if cdf_losses is not None:
            losses["cdf"] = cdf_losses
        if pdf_losses is not None:
            losses["pdf"] = pdf_losses
        return losses

    def _dual_losses(
        self,
        predictions: tensor_dict_type,
        is_synthetic: bool,
    ) -> tensor_dict_type:
        # median recover
        median_recover_losses = None
        if not is_synthetic:
            median_inverse = predictions["median_inverse"]
            median_recover_losses = torch.abs(median_inverse - 0.5)
            median_recover_losses = self._lb_recover * median_recover_losses
        # q recover
        q_batch = predictions["q_batch"]
        q_inverse = predictions["q_inverse"]
        q_recover_losses = l1_loss(q_inverse, q_batch, reduction="none")
        q_recover_losses = self._lb_recover * q_recover_losses
        # y recover
        y_batch = predictions["y_batch"]
        y_recover_losses = None
        if not is_synthetic:
            y_inverse = predictions["y_inverse_res"] + predictions["median"].detach()
            if y_inverse is not None:
                y_recover_losses = l1_loss(y_inverse, y_batch, reduction="none")
                y_recover_losses = self._lb_recover * y_recover_losses
        # combine
        losses = {"q_recover": q_recover_losses}
        if median_recover_losses is not None:
            losses["median_recover"] = median_recover_losses
        if y_recover_losses is not None:
            losses["y_recover"] = y_recover_losses
        return losses

    def _core(  # type: ignore
        self,
        predictions: tensor_dict_type,
        target: torch.Tensor,
        *,
        is_synthetic: bool = False,
    ) -> Tuple[torch.Tensor, tensor_dict_type]:
        losses = {}
        if self.fetch_q:
            losses.update(self._q_losses(predictions, target, is_synthetic))
        if self.fetch_cdf:
            losses.update(self._y_losses(predictions, target, is_synthetic))
        if self.fetch_q and self.fetch_cdf:
            losses.update(self._dual_losses(predictions, is_synthetic))
        suffix = "" if not is_synthetic else "synthetic_"
        losses = {f"{suffix}{k}": v for k, v in losses.items()}
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
        residual: torch.Tensor,
        target_residual: torch.Tensor,
        q_batch: torch.Tensor,
    ) -> torch.Tensor:
        quantile_error = target_residual - residual
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
