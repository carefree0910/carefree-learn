import torch

from typing import *
from torch.nn.functional import l1_loss
from torch.nn.functional import softplus

from ...losses import LossBase
from ...types import tensor_dict_type
from ...misc.toolkit import LoggingMixinWithRank
from ...modules.auxiliary import MTL


@LossBase.register("ddr")
class DDRLoss(LossBase, LoggingMixinWithRank):
    def _init_config(self, config: Dict[str, Any]) -> None:
        self.fetch_q = config["fetch_q"]
        self.fetch_cdf = config["fetch_cdf"]
        self.mtl = MTL(18, config["mtl_method"])
        self._lb_pdf = config.setdefault("lambda_pdf", 0.01)
        self._pdf_eps = config.setdefault("pdf_eps", 1.0e-12)
        self._lb_recover = config.setdefault("lambda_recover", 1.0)

    @staticmethod
    def _median_losses(
        forward_results: tensor_dict_type,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, tensor_dict_type]:
        # median
        median = forward_results["predictions"]
        median_losses = l1_loss(median, target, reduction="none")
        # median residual
        pos_med_res = forward_results["med_pos_med_res"]
        neg_med_res = forward_results["med_neg_med_res"]
        target_median_residual = target - median.detach()
        tmr_sign_mask = torch.sign(target_median_residual) > 0
        mr_prediction = torch.where(tmr_sign_mask, pos_med_res, -neg_med_res)
        mr_losses = torch.abs(target_median_residual - mr_prediction)
        losses_dict = {"median": median_losses, "median_residual": mr_losses}
        return target_median_residual, losses_dict

    def _q_losses(
        self,
        forward_results: tensor_dict_type,
        target_median_residual: Optional[torch.Tensor],
        is_synthetic: bool,
    ) -> tensor_dict_type:
        # median residual anchor
        if is_synthetic:
            syn_med_mul = forward_results["syn_med_mul"].abs()
            return {"median_residual_anchor": (syn_med_mul.abs() - 1.0).abs()}
        # quantile
        q_batch = forward_results["q_batch"]
        assert q_batch is not None
        y_res = forward_results["y_res"]
        assert y_res is not None and target_median_residual is not None
        quantile_losses = self._quantile_losses(y_res, target_median_residual, q_batch)
        return {"quantile": quantile_losses}

    def _y_losses(
        self,
        forward_results: tensor_dict_type,
        target: torch.Tensor,
        is_synthetic: bool,
    ) -> tensor_dict_type:
        # median residual anchor
        if is_synthetic:
            syn_cdf_logit_mul = forward_results["syn_cdf_logit_mul"]
            syn_med_cdf_logit_mul = forward_results["syn_med_cdf_logit_mul"]
            res_losses = (syn_cdf_logit_mul.abs() - 1.0).abs()
            med_losses = syn_med_cdf_logit_mul.abs()
            return {"cdf_anchor": res_losses + med_losses}
        # cdf
        y_batch = forward_results["y_batch"]
        cdf_logit = forward_results["cdf_logit"]
        assert y_batch is not None and cdf_logit is not None
        cdf_losses = self._cdf_losses(cdf_logit, target, y_batch)
        # pdf
        pdf = forward_results["pdf"]
        pdf_losses = None if pdf is None else self._pdf_losses(pdf)
        # combine
        losses = {"cdf": cdf_losses}
        if pdf_losses is not None:
            losses["pdf"] = pdf_losses
        return losses

    def _dual_losses(
        self,
        forward_results: tensor_dict_type,
        is_synthetic: bool,
    ) -> tensor_dict_type:
        # q recover
        q_batch = forward_results["q_batch"]
        q_inverse = forward_results["q_inverse"]
        q_recover_losses = l1_loss(q_inverse, q_batch, reduction="none")
        q_recover_losses = self._lb_recover * q_recover_losses
        # y recover
        y_batch = forward_results["y_batch"]
        y_recover_losses = None
        if not is_synthetic:
            median = forward_results["median"].detach()
            y_inverse = forward_results["y_inverse_res"] + median
            if y_inverse is not None:
                y_recover_losses = l1_loss(y_inverse, y_batch, reduction="none")
                y_recover_losses = self._lb_recover * y_recover_losses
        # combine
        losses = {"q_recover": q_recover_losses}
        if y_recover_losses is not None:
            losses["y_recover"] = y_recover_losses
        return losses

    def _core(  # type: ignore
        self,
        forward_results: tensor_dict_type,
        target: torch.Tensor,
        *,
        is_synthetic: bool = False,
    ) -> Tuple[torch.Tensor, tensor_dict_type]:
        losses: tensor_dict_type
        if is_synthetic:
            tmr, losses = None, {}
        else:
            tmr, losses = self._median_losses(forward_results, target)
        if self.fetch_q:
            losses.update(self._q_losses(forward_results, tmr, is_synthetic))
        if self.fetch_cdf:
            losses.update(self._y_losses(forward_results, target, is_synthetic))
        if self.fetch_q and self.fetch_cdf:
            losses.update(self._dual_losses(forward_results, is_synthetic))
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

    def _pdf_losses(self, pdf: torch.Tensor) -> torch.Tensor:
        negative_mask = pdf <= self._pdf_eps
        losses = torch.zeros_like(pdf)
        losses[negative_mask] = -pdf[negative_mask]
        positive_mask = ~negative_mask
        losses[positive_mask] = -self._lb_pdf * torch.log(pdf[positive_mask])
        return losses


__all__ = ["DDRLoss"]
