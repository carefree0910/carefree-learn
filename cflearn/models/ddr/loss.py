import torch
import logging

import torch.nn as nn

from typing import *
from cftool.ml import Anneal
from cftool.misc import LoggingMixin

from ...misc.toolkit import tensor_dict_type
from ...modules.auxiliary import MTL


class DDRLoss(nn.Module, LoggingMixin):
    def __init__(self,
                 config: Dict[str, Any],
                 device: torch.device):
        super().__init__()
        self._joint_training = config["joint_training"]
        self._use_dynamic_dual_loss_weights = config["use_dynamic_weights"]
        self._use_anneal, self._anneal_step = config["use_anneal"], config["anneal_step"]
        self._median_pressure = config.setdefault("median_pressure", 3.)
        self._median_pressure_inv = 1. / self._median_pressure
        self.mtl = MTL(16, config["mtl_method"])
        self._target_loss_warned = False
        self._zero = torch.zeros([1], dtype=torch.float32).to(device)
        if self._use_anneal:
            anneal_config = config.setdefault("anneal_config", {})
            anneal_methods = anneal_config.setdefault("methods", {})
            anneal_ratios = anneal_config.setdefault("ratios", {})
            anneal_floors = anneal_config.setdefault("floors", {})
            anneal_ceilings = anneal_config.setdefault("ceilings", {})
            default_anneal_methods = {
                "median_anneal": "linear", "main_anneal": "linear",
                "monotonous_anneal": "sigmoid", "anchor_anneal": "linear",
                "dual_anneal": "sigmoid", "recover_anneal": "sigmoid", "pressure_anneal": "sigmoid"
            }
            default_anneal_ratios = {
                "median_anneal": 0.25, "main_anneal": 0.25,
                "monotonous_anneal": 0.2, "anchor_anneal": 0.2,
                "dual_anneal": 0.75, "recover_anneal": 0.75, "pressure_anneal": 0.5
            }
            default_anneal_floors = {
                "median_anneal": 1., "main_anneal": 0.,
                "monotonous_anneal": 0., "anchor_anneal": 0.,
                "dual_anneal": 0., "recover_anneal": 0., "pressure_anneal": 0.
            }
            default_anneal_ceilings = {
                "median_anneal": 2.5, "main_anneal": 0.8,
                "monotonous_anneal": 2.5, "anchor_anneal": 2.,
                "dual_anneal": 0.1, "recover_anneal": 0.1, "pressure_anneal": 1.,
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
                    setattr(self, attr, Anneal(
                        anneal_methods[anneal], round(self._anneal_step * anneal_ratios[anneal]),
                        anneal_floors[anneal], anneal_ceilings[anneal]
                    ))

    def forward(self,
                predictions: tensor_dict_type,
                target: torch.Tensor,
                *,
                check_monotonous_only: bool = False) -> Tuple[torch.Tensor, tensor_dict_type]:
        # anneal
        if not self._use_anneal or not self.training or check_monotonous_only:
            main_anneal = median_anneal = None
            monotonous_anneal = anchor_anneal = None
            dual_anneal = recover_anneal = pressure_anneal = None
        else:
            main_anneal = None if self._main_anneal is None else self._main_anneal.pop()
            median_anneal = None if self._median_anneal is None else self._median_anneal.pop()
            monotonous_anneal = None if self._monotonous_anneal is None else self._monotonous_anneal.pop()
            anchor_anneal = None if self._median_anneal is None else self._anchor_anneal.pop()
            dual_anneal = None if self._median_anneal is None else self._dual_anneal.pop()
            recover_anneal = None if self._median_anneal is None else self._recover_anneal.pop()
            pressure_anneal = None if self._pressure_anneal is None else self._pressure_anneal.pop()
            self._last_main_anneal, self._last_pressure_anneal = main_anneal, pressure_anneal
        if self._use_anneal and check_monotonous_only:
            main_anneal, pressure_anneal = self._last_main_anneal, self._last_pressure_anneal
        # median
        median = predictions["predictions"]
        median_loss = nn.functional.l1_loss(median, target)
        if median_anneal is not None:
            median_loss = median_loss * median_anneal
        # get
        anchor_batch, cdf_raw = map(predictions.get, ["anchor_batch", "cdf_raw"])
        sampled_anchors, sampled_cdf_raw = map(predictions.get, ["sampled_anchors", "sampled_cdf_raw"])
        quantile_batch, median_residual, quantile_residual, quantile_sign = map(
            predictions.get, ["quantile_batch", "median_residual", "quantile_residual", "quantile_sign"])
        sampled_quantiles, sampled_quantile_residual = map(
            predictions.get, ["sampled_quantiles", "sampled_quantile_residual"])
        cdf_gradient, quantile_residual_gradient = map(
            predictions.get, ["cdf_gradient", "quantile_residual_gradient"])
        dual_quantile, quantile_cdf_raw = map(predictions.get, ["dual_quantile", "quantile_cdf_raw"])
        dual_cdf, cdf_quantile_residual = map(predictions.get, ["dual_cdf", "cdf_quantile_residual"])
        # cdf
        fetch_cdf = cdf_raw is not None
        cdf_anchor_loss = cdf_monotonous_loss = None
        if not fetch_cdf or check_monotonous_only:
            cdf_loss = cdf_losses = None
        else:
            cdf_losses = self._get_cdf_loss(target, cdf_raw, anchor_batch, False)
            if main_anneal is not None:
                cdf_losses = cdf_losses * main_anneal
            cdf_loss = cdf_losses.mean()
            if sampled_cdf_raw is not None:
                cdf_anchor_loss = self._get_cdf_loss(target, sampled_cdf_raw, sampled_anchors, True)
                if anchor_anneal is not None:
                    cdf_anchor_loss = cdf_anchor_loss * anchor_anneal
        # cdf monotonous
        if cdf_gradient is not None:
            cdf_monotonous_loss = nn.functional.relu(-cdf_gradient).mean()
            if anchor_anneal is not None:
                cdf_monotonous_loss = cdf_monotonous_loss * monotonous_anneal
        # quantile
        fetch_quantile = quantile_residual is not None
        quantile_anchor_loss = quantile_monotonous_loss = None
        if not fetch_quantile or check_monotonous_only:
            median_residual_loss = quantile_loss = quantile_losses = None
        else:
            target_median_residual = target - predictions["median_detach"]
            median_residual_loss = self._get_median_residual_loss(
                target_median_residual, median_residual, quantile_sign)
            if anchor_anneal is not None:
                median_residual_loss = median_residual_loss * anchor_anneal
            quantile_losses = self._get_quantile_residual_loss(
                target_median_residual, quantile_residual, quantile_batch, False)
            quantile_loss = quantile_losses.mean() + median_residual_loss
            if main_anneal is not None:
                quantile_loss = quantile_loss * main_anneal
            if sampled_quantile_residual is not None:
                quantile_anchor_loss = self._get_quantile_residual_loss(
                    target_median_residual, sampled_quantile_residual,
                    sampled_quantiles, True
                )
                if anchor_anneal is not None:
                    quantile_anchor_loss = quantile_anchor_loss * anchor_anneal
        # median pressure
        if not fetch_quantile:
            median_pressure_loss = None
        else:
            median_pressure_loss = self._get_median_pressure_loss(predictions)
            if pressure_anneal is not None:
                median_pressure_loss = median_pressure_loss * pressure_anneal
        # quantile monotonous
        quantile_monotonous_losses = []
        if quantile_residual_gradient is not None:
            quantile_monotonous_losses.append(nn.functional.relu(-quantile_residual_gradient).mean())
        if median_residual is not None and quantile_sign is not None:
            quantile_monotonous_losses.append(
                self._get_median_residual_monotonous_loss(median_residual, quantile_sign))
        if quantile_monotonous_losses:
            quantile_monotonous_loss = sum(quantile_monotonous_losses)
            if anchor_anneal is not None:
                quantile_monotonous_loss = quantile_monotonous_loss * monotonous_anneal
        # dual
        if not self._joint_training or not fetch_cdf or not fetch_quantile or check_monotonous_only:
            dual_cdf_loss = dual_quantile_loss = None
            cdf_recover_loss = quantile_recover_loss = None
        else:
            # dual cdf (cdf -> quantile [recover loss] -> cdf [dual loss])
            quantile_recover_loss, quantile_recover_losses, quantile_recover_loss_weights = \
                self._get_dual_recover_loss(dual_quantile, anchor_batch, cdf_losses)
            if quantile_cdf_raw is None:
                dual_quantile_loss = None
            else:
                dual_quantile_losses = self._get_cdf_loss(target, quantile_cdf_raw, anchor_batch, False)
                if quantile_recover_losses is None or not self._use_dynamic_dual_loss_weights:
                    dual_quantile_loss_weights = 1.
                else:
                    quantile_recover_losses_detach = quantile_recover_losses.detach()
                    dual_quantile_loss_weights = 0.5 * (
                        quantile_recover_loss_weights + 1 / (1 + 2 * torch.tanh(quantile_recover_losses_detach)))
                dual_quantile_loss = (dual_quantile_losses * dual_quantile_loss_weights).mean()
            # dual quantile (quantile -> cdf [recover loss] -> quantile [dual loss])
            cdf_recover_loss, cdf_recover_losses, cdf_recover_loss_weights = \
                self._get_dual_recover_loss(dual_cdf, quantile_batch, quantile_losses)
            if cdf_quantile_residual is None:
                dual_cdf_loss = None
            else:
                dual_cdf_losses = self._get_quantile_residual_loss(
                    target, cdf_quantile_residual, quantile_batch, False)
                if cdf_recover_losses is None or not self._use_dynamic_dual_loss_weights:
                    dual_cdf_loss_weights = 1.
                else:
                    cdf_recover_losses_detach = cdf_recover_losses.detach()
                    dual_cdf_loss_weights = 0.5 * (
                        cdf_recover_loss_weights + 1 / (1 + 10 * cdf_recover_losses_detach))
                dual_cdf_loss = (dual_cdf_losses * dual_cdf_loss_weights).mean() + median_residual_loss
        if dual_anneal is not None:
            if dual_cdf_loss is not None:
                dual_cdf_loss = dual_cdf_loss * dual_anneal
            if dual_quantile_loss is not None:
                dual_quantile_loss = dual_quantile_loss * dual_anneal
        if recover_anneal is not None:
            if cdf_recover_loss is not None:
                cdf_recover_loss = cdf_recover_loss * recover_anneal
            if quantile_recover_loss is not None:
                quantile_recover_loss = quantile_recover_loss * recover_anneal
        # combine
        if check_monotonous_only:
            losses = {}
        else:
            losses = {"median": median_loss}
            if not self._joint_training:
                if cdf_anchor_loss is not None:
                    losses["cdf_anchor"] = cdf_anchor_loss
                if quantile_anchor_loss is not None:
                    losses["quantile_anchor"] = quantile_anchor_loss
            else:
                if fetch_cdf:
                    losses["cdf"] = cdf_loss
                    if cdf_anchor_loss is not None:
                        losses["cdf_anchor"] = cdf_anchor_loss
                if fetch_quantile:
                    losses["quantile"] = quantile_loss
                    if quantile_anchor_loss is not None:
                        losses["quantile_anchor"] = quantile_anchor_loss
                if fetch_cdf and fetch_quantile:
                    losses["quantile_recover"], losses["cdf_recover"] = quantile_recover_loss, cdf_recover_loss
                    losses["dual_quantile"], losses["dual_cdf"] = dual_quantile_loss, dual_cdf_loss
        if median_residual_loss is not None:
            losses["median_residual_loss"] = median_residual_loss
        if median_pressure_loss is not None:
            key = "synthetic_median_pressure_loss" if check_monotonous_only else "median_pressure_loss"
            losses[key] = median_pressure_loss
        if cdf_monotonous_loss is not None:
            key = "synthetic_cdf_monotonous" if check_monotonous_only else "cdf_monotonous"
            losses[key] = cdf_monotonous_loss
        if quantile_monotonous_loss is not None:
            key = "synthetic_quantile_monotonous" if check_monotonous_only else "quantile_monotonous"
            losses[key] = quantile_monotonous_loss
        if not losses:
            return self._zero, {"loss": self._zero}
        if not self.mtl.registered:
            self.mtl.register(losses.keys())
        return self.mtl(losses), losses

    def _get_dual_recover_loss(self, dual_prediction, another_input_batch, another_losses):
        if dual_prediction is None:
            recover_loss = recover_losses = recover_loss_weights = None
        else:
            recover_losses = torch.abs(another_input_batch - dual_prediction)
            if not self._use_dynamic_dual_loss_weights:
                recover_loss_weights = 1.
            else:
                another_losses_detach = another_losses.detach()
                recover_loss_weights = 1 / (1 + 2 * torch.tanh(another_losses_detach))
            recover_loss = (recover_losses * recover_loss_weights).mean()
        return recover_loss, recover_losses, recover_loss_weights

    @staticmethod
    def _get_cdf_loss(target, cdf_raw, anchor_batch, reduce):
        indicative = (target <= anchor_batch).to(torch.float32)
        cdf_losses = -indicative * cdf_raw + nn.functional.softplus(cdf_raw)
        return cdf_losses if not reduce else cdf_losses.mean()

    @staticmethod
    def _get_median_residual_monotonous_loss(median_residual, quantile_sign):
        return nn.functional.relu(-median_residual * quantile_sign).mean()

    @staticmethod
    def _get_quantile_residual_loss(target_residual, quantile_residual, quantile_batch, reduce):
        quantile_error = target_residual - quantile_residual
        quantile_losses = torch.max(quantile_batch * quantile_error, (quantile_batch - 1) * quantile_error)
        return quantile_losses if not reduce else quantile_losses.mean()

    def _get_median_residual_loss(self, target_median_residual, median_residual, quantile_sign):
        same_sign_mask = quantile_sign * torch.sign(target_median_residual) > 0
        tmr, mr = map(lambda tensor: tensor[same_sign_mask], [target_median_residual, median_residual])
        median_residual_mae = self._median_pressure * torch.abs(tmr - mr).mean()
        residual_monotonous_loss = DDRLoss._get_median_residual_monotonous_loss(median_residual, quantile_sign)
        return median_residual_mae + residual_monotonous_loss

    def _get_median_pressure_loss(self, predictions):
        pressure_pos_dict, pressure_neg_dict = map(
            predictions.get, map(lambda attr: f"pressure_sub_quantile_{attr}_dict", ["pos", "neg"]))
        additive_pos, additive_neg = pressure_pos_dict["add"], pressure_neg_dict["add"]
        multiply_pos, multiply_neg = pressure_pos_dict["mul"], pressure_neg_dict["mul"]
        # additive net & multiply net are tend to be zero here
        # because median pressure batch receives 0.5 as input
        return sum(
            torch.max(
                -self._median_pressure * sub_quantile,
                self._median_pressure_inv * sub_quantile
            ).mean()
            for sub_quantile in [
                additive_pos, -additive_neg,
                multiply_pos, multiply_neg
            ]
        )


__all__ = ["DDRLoss"]
