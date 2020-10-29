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
        self._lb_std = config.setdefault("lambda_std", 1e-5)
        self._use_anneal = config["use_anneal"]
        self._anneal_step = config["anneal_step"]
        self.mtl = MTL(7, config["mtl_method"])
        self.register_buffer("zero", torch.zeros([1, 1], dtype=torch.float32))
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
                "main_anneal": 0.25,
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

    @staticmethod
    def _pdf_loss(pdf: torch.Tensor) -> torch.Tensor:
        negative_mask = pdf <= 1e-8
        monotonous_loss = torch.sum(-pdf[negative_mask])
        log_likelihood_loss = torch.sum(-torch.log(pdf[~negative_mask]))
        return (monotonous_loss + log_likelihood_loss) / len(pdf)

    def _core(  # type: ignore
        self,
        predictions: tensor_dict_type,
        target: torch.Tensor,
        *,
        check_monotonous_only: bool = False,
    ) -> Tuple[torch.Tensor, tensor_dict_type]:
        # anneal
        if not self._use_anneal or not self.training or check_monotonous_only:
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
        if self._use_anneal and check_monotonous_only:
            main_anneal = self._last_main_anneal
        # median
        median = predictions["predictions"]
        median_losses = l1_loss(median, target, reduction="none")
        if median_anneal is not None:
            median_losses = median_losses * median_anneal
        # quantile
        quantile_anchor_losses = None
        if check_monotonous_only:
            quantile_losses = None
        else:
            q_batch = predictions["q_batch"]
            quantiles = predictions["quantiles"]
            quantiles_full = predictions["quantiles_full"]
            assert q_batch is not None
            assert quantiles is not None
            assert quantiles_full is not None
            quantile_losses = self._get_quantile_losses(quantiles, target, q_batch)
            std_losses = self._lb_std * quantiles_full.std(1, keepdim=True)
            quantile_losses = quantile_losses + std_losses
            if main_anneal is not None:
                quantile_losses = quantile_losses * main_anneal
            sampled_q_batch = predictions["sampled_q_batch"]
            if sampled_q_batch is not None:
                sampled_quantiles = predictions["sampled_quantiles"]
                sampled_quantiles_full = predictions["sampled_quantiles_full"]
                assert sampled_quantiles is not None
                assert sampled_quantiles_full is not None
                quantile_anchor_losses = self._get_quantile_losses(
                    sampled_quantiles,
                    target,
                    sampled_q_batch,
                )
                std_losses = self._lb_std * sampled_quantiles_full.std(1, keepdim=True)
                quantile_anchor_losses = quantile_anchor_losses + std_losses
                if anchor_anneal is not None:
                    quantile_anchor_losses = quantile_anchor_losses * anchor_anneal
        # cdf
        cdf_anchor_losses = None
        if check_monotonous_only:
            cdf_losses = None
        else:
            cdf = predictions["cdf"]
            cdf_full = predictions["cdf_full"]
            y_batch = predictions["y_batch"]
            assert cdf is not None
            assert cdf_full is not None
            assert y_batch is not None
            cdf_losses = self._get_cdf_losses(cdf, target, y_batch)
            std_losses = self._lb_std * cdf_full.std(1, keepdim=True)
            cdf_losses = cdf_losses + std_losses
            if main_anneal is not None:
                cdf_losses = cdf_losses * main_anneal
            sampled_y_batch = predictions["sampled_y_batch"]
            if sampled_y_batch is not None:
                sampled_cdf = predictions["sampled_cdf"]
                sampled_cdf_full = predictions["sampled_cdf_full"]
                assert sampled_cdf is not None
                assert sampled_cdf_full is not None
                cdf_anchor_losses = self._get_cdf_losses(
                    sampled_cdf,
                    target,
                    sampled_y_batch,
                )
                std_losses = self._lb_std * sampled_cdf_full.std(1, keepdim=True)
                cdf_anchor_losses = cdf_anchor_losses + std_losses
                if anchor_anneal is not None:
                    cdf_anchor_losses = cdf_anchor_losses * anchor_anneal
        # pdf
        pdf_losses = None
        pdf, sampled_pdf = map(predictions.get, ["pdf", "sampled_pdf"])
        if pdf is not None and sampled_pdf is not None:
            pdf_losses = self._pdf_loss(pdf) + self._pdf_loss(sampled_pdf)
            if anchor_anneal is not None:
                pdf_losses = pdf_losses * monotonous_anneal
        # combine
        if check_monotonous_only:
            losses = {}
        else:
            losses = {"median": median_losses}
            assert cdf_losses is not None
            losses["cdf"] = cdf_losses
            if cdf_anchor_losses is not None:
                losses["cdf_anchor"] = cdf_anchor_losses
            assert quantile_losses is not None
            losses["quantile"] = quantile_losses
            if quantile_anchor_losses is not None:
                losses["quantile_anchor"] = quantile_anchor_losses
        if pdf_losses is not None:
            key = "synthetic_pdf" if check_monotonous_only else "pdf"
            losses[key] = pdf_losses
        # TODO : check this condition branch and see if it is necessary
        if not losses:
            zero = self.zero.repeat(len(target), 1)  # type: ignore
            return zero, {"loss": zero}
        if not self.mtl.registered:
            self.mtl.register(list(losses.keys()))
        return self.mtl(losses), losses

    def forward(  # type: ignore
        self,
        predictions: tensor_dict_type,
        target: torch.Tensor,
        *,
        check_monotonous_only: bool = False,
    ) -> Tuple[torch.Tensor, tensor_dict_type]:
        losses, losses_dict = self._core(
            predictions,
            target,
            check_monotonous_only=check_monotonous_only,
        )
        reduced_losses = self._reduce(losses)
        reduced_losses_dict = {k: self._reduce(v) for k, v in losses_dict.items()}
        return reduced_losses, reduced_losses_dict

    @staticmethod
    def _get_cdf_losses(
        cdf: torch.Tensor,
        target: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> torch.Tensor:
        mask = target <= y_batch
        mask, rev_mask = mask.to(torch.float32), (~mask).to(torch.float32)
        return -(mask * torch.log(cdf) + rev_mask * torch.log(1.0 - cdf))

    @staticmethod
    def _get_quantile_losses(
        quantiles: torch.Tensor,
        target: torch.Tensor,
        q_batch: torch.Tensor,
    ) -> torch.Tensor:
        quantile_error = target - quantiles
        q1 = q_batch * quantile_error
        q2 = (q_batch - 1.0) * quantile_error
        return torch.max(q1, q2)


__all__ = ["DDRLoss"]
