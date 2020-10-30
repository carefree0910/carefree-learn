import torch

from typing import *
from cftool.ml import Anneal
from cftool.misc import LoggingMixin
from torch.nn.functional import l1_loss


from ...losses import LossBase
from ...types import tensor_dict_type
from ...misc.toolkit import to_numpy
from ...modules.auxiliary import MTL


class DDRLoss(LossBase, LoggingMixin):
    def _init_config(self, config: Dict[str, Any]) -> None:
        self._lb_recover = config.setdefault("lambda_recover", 10.0)
        self._use_anneal = False
        # self._use_anneal = config["use_anneal"]
        self._anneal_step = config["anneal_step"]
        self.mtl = MTL(18, config["mtl_method"])
        if self._use_anneal:
            self._median_anneal: Anneal
            self._main_anneal: Anneal
            self._monotonous_anneal: Anneal
            self._anchor_anneal: Anneal
            anneal_config = config.setdefault("anneal_config", {})
            anneal_methods = anneal_config.setdefault("methods", {})
            anneal_ratios = anneal_config.setdefault("ratios", {})
            anneal_floors = anneal_config.setdefault("floors", {})
            anneal_ceilings = anneal_config.setdefault("ceilings", {})
            default_anneal_methods = {
                "median_anneal": "linear",
                "main_anneal": "linear",
                "monotonous_anneal": "sigmoid",
                "anchor_anneal": "linear",
            }
            default_anneal_ratios = {
                "median_anneal": 0.25,
                "main_anneal": 0.25,
                "monotonous_anneal": 0.2,
                "anchor_anneal": 0.2,
            }
            default_anneal_floors = {
                "median_anneal": 1.0,
                "main_anneal": 0.0,
                "monotonous_anneal": 0.0,
                "anchor_anneal": 0.0,
            }
            default_anneal_ceilings = {
                "median_anneal": 2.5,
                "main_anneal": 1.0,
                "monotonous_anneal": 2.5,
                "anchor_anneal": 2.0,
            }
            for anneal in default_anneal_methods:
                anneal_methods.setdefault(anneal, default_anneal_methods[anneal])
                anneal_ratios.setdefault(anneal, default_anneal_ratios[anneal])
                anneal_floors.setdefault(anneal, default_anneal_floors[anneal])
                anneal_ceilings.setdefault(anneal, default_anneal_ceilings[anneal])
            for anneal in default_anneal_methods:
                attr = f"_{anneal}"
                if anneal_methods[anneal] is None:
                    setattr(self, attr, None)
                else:
                    setattr(
                        self,
                        attr,
                        Anneal(
                            anneal_methods[anneal],
                            round(self._anneal_step * anneal_ratios[anneal]),
                            anneal_floors[anneal],
                            anneal_ceilings[anneal],
                        ),
                    )

    def _core(  # type: ignore
        self,
        predictions: tensor_dict_type,
        target: torch.Tensor,
        *,
        is_synthetic: bool = False,
    ) -> Tuple[torch.Tensor, tensor_dict_type]:
        # anneal
        if not self._use_anneal or not self.training or is_synthetic:
            main_anneal = median_anneal = None
            monotonous_anneal = anchor_anneal = None
        else:
            main_anneal = None if self._main_anneal is None else self._main_anneal.pop()
            median_anneal = (
                None if self._median_anneal is None else self._median_anneal.pop()
            )
            monotonous_anneal = (
                None
                if self._monotonous_anneal is None
                else self._monotonous_anneal.pop()
            )
            anchor_anneal = (
                None if self._median_anneal is None else self._anchor_anneal.pop()
            )
            self._last_main_anneal = main_anneal
        if self._use_anneal and is_synthetic:
            main_anneal = self._last_main_anneal
        # median
        if is_synthetic:
            median_losses = median_recover_losses = None
        else:
            median = predictions["predictions"]
            median_inverse = predictions["median_inverse"]
            median_losses = l1_loss(median, target, reduction="none")
            median_recover_losses = torch.abs(median_inverse - 0.5)
            median_recover_losses = self._lb_recover * median_recover_losses
        if median_anneal is not None:
            median_losses = median_losses * median_anneal
            median_recover_losses = median_recover_losses * median_anneal
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
            if main_anneal is not None:
                quantile_losses = quantile_losses * main_anneal
            sampled_q_batch = predictions["sampled_q_batch"]
            if sampled_q_batch is not None:
                sampled_y = predictions["sampled_y"]
                assert sampled_y is not None
                anchor_quantile_losses = self._quantile_losses(
                    sampled_y,
                    target,
                    sampled_q_batch,
                )
                if anchor_anneal is not None:
                    anchor_quantile_losses = anchor_quantile_losses * anchor_anneal
        # q recover losses
        q_inverse = predictions["q_inverse"]
        assert q_inverse is not None
        q_recover_losses = l1_loss(q_inverse, q_batch, reduction="none")
        q_recover_losses = self._lb_recover * q_recover_losses
        if main_anneal is not None:
            q_recover_losses = q_recover_losses * main_anneal
        sampled_q_batch = predictions["sampled_q_batch"]
        aq_recover_losses = None
        if sampled_q_batch is not None:
            sq_inverse = predictions["sampled_q_inverse"]
            assert sq_inverse is not None
            aq_recover_losses = l1_loss(sq_inverse, sampled_q_batch, reduction="none")
            aq_recover_losses = self._lb_recover * aq_recover_losses
            if anchor_anneal is not None:
                aq_recover_losses = aq_recover_losses * anchor_anneal
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
            if main_anneal is not None:
                cdf_losses = cdf_losses * main_anneal
            sampled_y_batch = predictions["sampled_y_batch"]
            if sampled_y_batch is not None:
                sampled_cdf = predictions["sampled_cdf"]
                assert sampled_cdf is not None
                anchor_cdf_losses = self._cdf_losses(
                    sampled_cdf,
                    target,
                    sampled_y_batch,
                )
                if anchor_anneal is not None:
                    anchor_cdf_losses = anchor_cdf_losses * anchor_anneal
        # y recover losses
        ay_recover_losses = None
        y_inverse = predictions["y_inverse"]
        assert y_inverse is not None
        y_recover_losses = l1_loss(y_inverse, y_batch, reduction="none")
        y_recover_losses = self._lb_recover * y_recover_losses
        if main_anneal is not None:
            y_recover_losses = y_recover_losses * main_anneal
        sampled_y_batch = predictions["sampled_y_batch"]
        if sampled_y_batch is not None:
            sy_inverse = predictions["sampled_y_inverse"]
            assert sy_inverse is not None
            ay_recover_losses = l1_loss(sy_inverse, sampled_y_batch, reduction="none")
            ay_recover_losses = self._lb_recover * ay_recover_losses
            if anchor_anneal is not None:
                ay_recover_losses = ay_recover_losses * anchor_anneal
        # pdf losses
        pdf, sampled_pdf = map(predictions.get, ["pdf", "sampled_pdf"])
        if pdf is None and sampled_pdf is None:
            pdf_losses = None
        elif pdf is not None and sampled_pdf is not None:
            pdf_losses = self._pdf_losses(pdf) + self._pdf_losses(sampled_pdf)
        elif pdf is not None:
            pdf_losses = self._pdf_losses(pdf)
        else:
            pdf_losses = self._pdf_losses(sampled_pdf)
        if pdf_losses is not None and anchor_anneal is not None:
            pdf_losses = pdf_losses * monotonous_anneal
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

    @staticmethod
    def _cdf_losses(
        cdf: torch.Tensor,
        target: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> torch.Tensor:
        mask = target <= y_batch
        mask, rev_mask = mask.to(torch.float32), (~mask).to(torch.float32)
        return -(mask * torch.log(cdf) + rev_mask * torch.log(1.0 - cdf))

    @staticmethod
    def _pdf_losses(pdf: torch.Tensor) -> torch.Tensor:
        negative_mask = pdf <= 1e-8
        positive_mask = ~negative_mask
        losses = torch.zeros_like(pdf)
        losses[negative_mask] = -pdf[negative_mask]
        losses[positive_mask] = -torch.log(pdf[positive_mask])
        return losses


__all__ = ["DDRLoss"]
