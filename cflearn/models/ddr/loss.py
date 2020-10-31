import torch

from typing import *
from cftool.misc import LoggingMixin
from torch.nn.functional import l1_loss


from ...losses import LossBase
from ...types import tensor_dict_type
from ...modules.auxiliary import MTL


class DDRLoss(LossBase, LoggingMixin):
    def _init_config(self, config: Dict[str, Any]) -> None:
        self.mtl = MTL(18, config["mtl_method"])
        self._cdf_floor = config.setdefault("cdf_eps", 1.0e-8)
        self._cdf_ceiling = 1.0 - self._cdf_floor
        self._lb_recover = config.setdefault("lambda_recover", 1.0)

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
            median_inverse = predictions["median_inverse"]
            median_losses = l1_loss(median, target, reduction="none")
            median_recover_losses = torch.abs(median_inverse - 0.5)
            median_recover_losses = self._lb_recover * median_recover_losses
        # quantile losses
        q_batch = predictions["q_batch"]
        assert q_batch is not None
        anchor_quantile_losses = None
        if is_synthetic:
            quantile_losses = None
        else:
            y = predictions["y"]
            assert y is not None
            quantile_losses = self._quantile_losses(y, target, q_batch)
            sampled_q_batch = predictions["sampled_q_batch"]
            if sampled_q_batch is not None:
                sampled_y = predictions["sampled_y"]
                assert sampled_y is not None
                anchor_quantile_losses = self._quantile_losses(
                    sampled_y,
                    target,
                    sampled_q_batch,
                )
        # q recover losses
        q_inverse = predictions["q_inverse"]
        assert q_inverse is not None
        q_recover_losses = l1_loss(q_inverse, q_batch, reduction="none")
        q_recover_losses = self._lb_recover * q_recover_losses
        sampled_q_batch = predictions["sampled_q_batch"]
        aq_recover_losses = None
        if sampled_q_batch is not None:
            sq_inverse = predictions["sampled_q_inverse"]
            assert sq_inverse is not None
            aq_recover_losses = l1_loss(sq_inverse, sampled_q_batch, reduction="none")
            aq_recover_losses = self._lb_recover * aq_recover_losses
        # cdf losses
        y_batch = predictions["y_batch"]
        assert y_batch is not None
        anchor_cdf_losses = None
        if is_synthetic:
            cdf_losses = None
        else:
            cdf = predictions["cdf"]
            assert cdf is not None
            cdf_losses = self._cdf_losses(cdf, target, y_batch)
            sampled_y_batch = predictions["sampled_y_batch"]
            if sampled_y_batch is not None:
                sampled_cdf = predictions["sampled_cdf"]
                assert sampled_cdf is not None
                anchor_cdf_losses = self._cdf_losses(
                    sampled_cdf,
                    target,
                    sampled_y_batch,
                )
        # y recover losses
        ay_recover_losses = None
        y_inverse = predictions["y_inverse"]
        assert y_inverse is not None
        y_recover_losses = l1_loss(y_inverse, y_batch, reduction="none")
        y_recover_losses = self._lb_recover * y_recover_losses
        sampled_y_batch = predictions["sampled_y_batch"]
        if sampled_y_batch is not None:
            sy_inverse = predictions["sampled_y_inverse"]
            assert sy_inverse is not None
            ay_recover_losses = l1_loss(sy_inverse, sampled_y_batch, reduction="none")
            ay_recover_losses = self._lb_recover * ay_recover_losses
        # pdf losses
        pdf, sampled_pdf = map(predictions.get, ["pdf", "sampled_pdf"])
        if pdf is None and sampled_pdf is None:
            pdf_losses = None
        elif pdf is not None and sampled_pdf is not None:
            pdf_losses = self._pdf_losses(pdf) + self._pdf_losses(sampled_pdf)
        elif pdf is not None:
            pdf_losses = self._pdf_losses(pdf)
        else:
            assert sampled_pdf is not None
            pdf_losses = self._pdf_losses(sampled_pdf)
        # combine
        losses = {}
        suffix = "" if not is_synthetic else "synthetic_"
        # q recover
        assert q_recover_losses is not None
        losses[f"{suffix}q_recover"] = q_recover_losses
        if aq_recover_losses is not None:
            losses[f"{suffix}anchor_q_recover"] = aq_recover_losses
        # y recover
        losses[f"{suffix}y_recover"] = y_recover_losses
        if ay_recover_losses is not None:
            losses[f"{suffix}anchor_y_recover"] = ay_recover_losses
        # common
        if not is_synthetic:
            assert median_losses is not None
            assert median_recover_losses is not None
            losses["median"] = median_losses
            losses["median_recover"] = median_recover_losses
            # quantile
            assert quantile_losses is not None
            losses["quantile"] = quantile_losses
            if anchor_quantile_losses is not None:
                losses["anchor_quantile"] = anchor_quantile_losses
            # cdf
            assert cdf_losses is not None
            losses["cdf"] = cdf_losses
            if anchor_cdf_losses is not None:
                losses["anchor_cdf"] = anchor_cdf_losses
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

    def _cdf_losses(
        self,
        cdf: torch.Tensor,
        target: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> torch.Tensor:
        mask = target <= y_batch
        cdf = torch.clamp(cdf, self._cdf_floor, self._cdf_ceiling)
        likelihood = torch.where(mask, torch.log(cdf), torch.log(1.0 - cdf))
        return -likelihood

    @staticmethod
    def _pdf_losses(pdf: torch.Tensor) -> torch.Tensor:
        negative_mask = pdf <= 1e-8
        losses = torch.zeros_like(pdf)
        losses[negative_mask] = -pdf[negative_mask]
        return losses


__all__ = ["DDRLoss"]
