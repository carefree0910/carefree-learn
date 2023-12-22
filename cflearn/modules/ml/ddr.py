import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from cftool.misc import print_warning
from cftool.types import tensor_dict_type

from .fcnn import FCNN
from ..common import register_module
from ...schema import losses_type
from ...schema import ILoss
from ...schema import TrainerState
from ...losses import register_loss
from ...toolkit import get_gradient
from ...toolkit import toggle_module
from ...constants import LOSS_KEY
from ...constants import LABEL_KEY
from ...constants import PREDICTIONS_KEY
from ..implicit.siren import make_grid
from ..implicit.siren import Siren


TCond = Union[float, List[float], np.ndarray, Tensor]


def all_exists(*tensors: Optional[Tensor]) -> bool:
    return all(tensor is not None for tensor in tensors)


def _expand_element(
    n: int,
    element: TCond,
    device: Optional[torch.device] = None,
) -> Tensor:
    if isinstance(element, Tensor):
        return element
    if isinstance(element, np.ndarray):
        element_arr = np.repeat(element[None], n, axis=0).astype(np.float32)
        element_arr = element_arr[..., None]
    elif isinstance(element, float):
        element_arr = np.repeat(element, n).astype(np.float32)
        element_arr = element_arr.reshape([-1, 1, 1])
    else:
        element_arr = np.array(element, np.float32).reshape([1, -1, 1])
        element_arr = np.repeat(element_arr, n, axis=0)
    element_tensor = torch.from_numpy(element_arr)
    if device is not None:
        element_tensor = element_tensor.to(device)
    return element_tensor


def _make_ddr_grid(num_samples: int, device: torch.device) -> Tensor:
    return make_grid(num_samples, 1, device)


@register_module("ddr")
class DDR(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: Optional[List[int]] = None,
        *,
        mapping_type: str = "highway",
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
        w_sin: float = 1.0,
        w_sin_initial: float = 30.0,
        num_random_samples: int = 16,
        use_extra_modulars: bool = False,
        predict_quantiles: bool = True,
        predict_cdf: bool = True,
        dual_period: Optional[int] = 2,
        use_dual_quantiles: bool = False,
        use_detached_dual_quantiles: bool = True,
        correction_period: Optional[int] = 2,
        use_mean_correction: bool = False,
        num_mean_samples: int = 64,
        y_min_max: Optional[Tuple[float, float]] = None,
    ):
        def _make_fcnn() -> FCNN:
            return FCNN(
                input_dim,
                output_dim,
                hidden_units,
                mapping_type=mapping_type,
                bias=bias,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )

        def _make_siren(_in_dim: Optional[int] = None) -> Siren:
            return Siren(
                None,
                _in_dim or 1,
                output_dim,
                hidden_units[0],  # type: ignore
                num_layers=len(hidden_units),  # type: ignore
                w_sin=w_sin,
                w_sin_initial=w_sin_initial,
                bias=False,
                use_modulator=False,
            )

        super().__init__()
        self.fcnn = _make_fcnn()
        hidden_units = self.fcnn.hidden_units
        assert hidden_units is not None
        if not len(set(hidden_units)) == 1:
            raise ValueError("`DDR` requires all hidden units to be identical")

        use_q_mod = predict_quantiles and use_extra_modulars
        use_cdf_mod = predict_cdf and use_extra_modulars
        self.q_mod = None if not use_q_mod else _make_fcnn()
        self.cdf_mod = None if not use_cdf_mod else _make_fcnn()

        self.dual_period = dual_period
        self.use_dual_quantiles = use_dual_quantiles
        self.use_detached_dual_quantiles = use_detached_dual_quantiles

        if use_mean_correction and not predict_quantiles:
            print_warning(
                "`use_mean_correction` is set to `True` but "
                "`predict_quantiles` is `False`, so `use_mean_correction` "
                "will fallback to `False`"
            )
            use_mean_correction = False
        self.correction_period = correction_period
        self.use_mean_correction = use_mean_correction
        self.num_mean_samples = num_mean_samples

        self.predict_quantiles = predict_quantiles
        self.predict_cdf = predict_cdf
        self.q_siren = None if not predict_quantiles else _make_siren()
        self.cdf_siren = None if not predict_cdf else _make_siren(output_dim)
        self.num_random_samples = num_random_samples
        self._y_min_max = y_min_max
        self.register_buffer("y_min_max", torch.tensor([0.0, 0.0]))

    def forward(
        self,
        net: Tensor,
        state: Optional[TrainerState] = None,
        *,
        get_quantiles: bool = True,
        get_cdf: bool = True,
        get_mean: Optional[bool] = None,
        tau: Optional[TCond] = None,
        y_anchor: Optional[TCond] = None,
    ) -> tensor_dict_type:
        # prepare
        device = net.device
        num_samples = len(net)
        if len(net.shape) > 2:
            net = net.contiguous().view(num_samples, -1)
        get_quantiles = get_quantiles and self.predict_quantiles
        get_cdf = get_cdf and self.predict_cdf
        if not self.training:
            if get_mean is None and get_quantiles and self.use_mean_correction:
                get_mean = True
        else:
            if (
                state is not None
                and self.use_mean_correction
                and (self.dual_period is None or state.step % self.dual_period == 0)
            ):
                get_mean = True
            else:
                get_mean = False
        if get_mean and not get_quantiles:
            print_warning(
                "`get_mean` is set to `True` but `get_quantiles` is `False`, "
                "so `get_mean` will fallback to `False`"
            )
            get_mean = False
        # median / modulator forward
        mods = []
        q_mods = []
        cdf_mods = []
        q_net = None
        cdf_net = None
        for i, block in enumerate(self.fcnn.net):
            is_last = i == len(self.fcnn.net) - 1
            if not is_last and self.q_mod is not None:
                q_net = net if q_net is None else q_net + net
            if not is_last and self.cdf_mod is not None:
                cdf_net = net if cdf_net is None else cdf_net + net
            net = block(net)
            mods.append(net)
            if not is_last:
                if self.q_mod is not None:
                    q_net = self.q_mod.net[i](q_net)
                    q_mods.append(q_net)
                if self.cdf_mod is not None:
                    cdf_net = self.cdf_mod.net[i](cdf_net)
                    cdf_mods.append(cdf_net)
        q_mods = q_mods or mods
        cdf_mods = cdf_mods or mods
        median = mods.pop()
        results = {PREDICTIONS_KEY: median}
        if not get_quantiles and not get_cdf:
            return results
        y_min, y_max = self.y_min_max.tolist()  # type: ignore
        y_span = y_max - y_min
        # quantile forward
        if get_quantiles:
            if tau is not None:
                tau_tensor = _expand_element(num_samples, tau, device) * 2.0 - 1.0
            else:
                if self.training:
                    shape = num_samples, self.num_random_samples, 1
                    tau_tensor = torch.rand(*shape, device=device) * 2.0 - 1.0
                else:
                    tau_tensor = _make_ddr_grid(self.num_random_samples, device)
                    tau_tensor = tau_tensor.repeat(num_samples, 1, 1)
            tau_tensor.requires_grad_(True)
            q_increment, quantiles = self._get_quantiles(tau_tensor, q_mods, median)
            results.update(
                {
                    "tau": tau_tensor,
                    "q_increment": q_increment,
                    "quantiles": quantiles,
                }
            )
            ## mean forward
            if get_mean:
                tau_mu = _make_ddr_grid(self.num_mean_samples, device)
                tau_mu = tau_mu.repeat(num_samples, 1, 1)
                _, quantiles_mu = self._get_quantiles(tau_mu, q_mods, median.detach())
                delta = 0.5 / self.num_mean_samples
                sliding_sum = quantiles_mu[:, 1:] + quantiles_mu[:, :-1]
                results["mean"] = delta * sliding_sum.sum(1)
        # cdf forward
        if get_cdf:
            if y_anchor is not None:
                ya_tensor = _expand_element(num_samples, y_anchor, device)
            else:
                shape = num_samples, self.num_random_samples, 1
                if self.training:
                    ya_tensor = torch.rand(*shape, device=device) * y_span + y_min
                else:
                    y_raw_ratio = _make_ddr_grid(self.num_random_samples, device)
                    y_raw_ratio = 0.5 * (y_raw_ratio + 1.0)
                    ya_tensor = (y_raw_ratio * y_span + y_min).repeat(num_samples, 1, 1)
            ya_tensor.requires_grad_(True)
            _, logit, cdf = self._get_cdf(ya_tensor, median, y_span, cdf_mods)
            pdf = get_gradient(cdf, ya_tensor, True, True)  # type: ignore
            results.update(
                {
                    "y_anchor": ya_tensor,
                    "logit": logit,
                    "cdf": cdf,
                    "pdf": pdf,
                }
            )
        # dual forward
        if (
            get_quantiles
            and get_cdf
            and (
                state is None
                or not self.training
                or self.dual_period is None
                or state.step % self.dual_period == 0
            )
        ):
            dual_y = results["quantiles"].detach()
            results["dual_cdf"] = self._get_cdf(dual_y, median, y_span, cdf_mods)[-1]
            if self.use_dual_quantiles:
                detach = self.use_detached_dual_quantiles
                dual_tau = results["cdf"] * 2.0 - 1.0
                if not detach:
                    dual_q_mods = q_mods
                else:
                    median = median.detach()
                    dual_q_mods = [mod.detach() for mod in q_mods]
                dual_q_args = dual_tau, dual_q_mods, median
                with toggle_module(self.q_siren, requires_grad=False, enabled=detach):
                    _, results["dual_quantiles"] = self._get_quantiles(*dual_q_args)
        return results

    # internal

    def _get_quantiles(
        self,
        tau: Tensor,
        mods: List[Tensor],
        median: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        q_increment = self.q_siren(mods, init=tau)  # type: ignore
        return q_increment, median[:, None] + q_increment

    def _get_cdf(
        self,
        y_anchor: Tensor,
        median: Tensor,
        y_span: float,
        mods: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        y_residual = y_anchor - median.detach().unsqueeze(1)
        y_ratio = y_residual / y_span
        logit = self.cdf_siren(mods, init=y_ratio)  # type: ignore
        cdf = torch.sigmoid(logit)
        return y_ratio, logit, cdf


@register_loss("ddr")
class DDRLoss(ILoss):
    def __init__(
        self,
        *,
        lb_ddr: float = 1.0,
        lb_dual: float = 1.0,
        lb_correction: float = 1.0,
        lb_monotonous: float = 1.0,
    ):
        super().__init__()
        self.lb_ddr = lb_ddr
        self.lb_dual = lb_dual
        self.lb_correction = lb_correction
        self.lb_monotonous = lb_monotonous

    def get_forward_args(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> Tuple[Any, ...]:
        return forward_results, batch

    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
    ) -> losses_type:
        tau = forward_results.get("tau")
        quantiles = forward_results.get("quantiles")
        q_increment = forward_results.get("q_increment")
        cdf = forward_results.get("cdf")
        pdf = forward_results.get("pdf")
        logit = forward_results.get("logit")
        y_anchor = forward_results.get("y_anchor")
        dual_cdf = forward_results.get("dual_cdf")
        dual_quantiles = forward_results.get("dual_quantiles")
        mean = forward_results.get("mean")
        median = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        losses = {}
        weighted_losses = []
        # mae
        mae = F.l1_loss(median, labels, reduction="none")
        losses["mae"] = mae
        weighted_losses.append(mae)
        # mean correction
        if all_exists(mean):
            mean_loss = F.mse_loss(mean, labels, reduction="none")
            losses["mse"] = mean_loss
            weighted_losses.append(self.lb_correction * mean_loss)
        # unsqueeze labels
        labels = labels[:, None]
        # quantiles
        if all_exists(tau, quantiles, q_increment):
            quantile_error = labels - quantiles
            tau_raw = 0.5 * (tau.detach() + 1.0)  # type: ignore
            neg_errors = tau_raw * quantile_error
            pos_errors = (tau_raw - 1.0) * quantile_error
            q_loss = torch.max(neg_errors, pos_errors).mean((1, 2), keepdim=True)
            g_tau = get_gradient(q_increment, tau, retain_graph=True, create_graph=True)  # type: ignore
            g_tau_loss = F.relu(-g_tau, inplace=True).mean((1, 2), keepdim=True)  # type: ignore
            losses["q"] = q_loss
            losses["g_tau"] = g_tau_loss
            weighted_losses.append(self.lb_ddr * q_loss)
            weighted_losses.append(self.lb_monotonous * g_tau_loss)
        # cdf
        if all_exists(cdf, pdf, logit, y_anchor):
            indicative = (labels <= y_anchor).to(torch.float32)
            cdf_mle = -indicative * logit + F.softplus(logit)
            cdf_mle = cdf_mle.mean((1, 2), keepdim=True)
            cdf_crps = (cdf - indicative) ** 2 * (labels - y_anchor).abs()
            cdf_crps = cdf_crps.mean((1, 2), keepdim=True)
            cdf_loss = cdf_mle + cdf_crps
            pdf_loss = F.relu(-pdf, inplace=True).mean((1, 2), keepdim=True)  # type: ignore
            losses["cdf"] = cdf_loss
            losses["cdf_mle"] = cdf_mle
            losses["cdf_crps"] = cdf_crps
            losses["pdf"] = pdf_loss
            weighted_losses.append(self.lb_ddr * cdf_loss)
            weighted_losses.append(self.lb_monotonous * pdf_loss)
        # dual
        if all_exists(dual_cdf):
            tau_raw = 0.5 * (tau.detach() + 1.0)  # type: ignore
            tau_recover_loss = F.l1_loss(tau_raw, dual_cdf)  # type: ignore
            losses["tau_recover"] = tau_recover_loss
            weighted_losses.append(self.lb_dual * tau_recover_loss)
            if dual_quantiles is not None:
                y_raw = y_anchor.detach()
                y_recover_loss = F.l1_loss(y_raw, dual_quantiles)
                losses["y_recover"] = y_recover_loss
                weighted_losses.append(self.lb_dual * y_recover_loss)
        # aggregate
        losses[LOSS_KEY] = sum(weighted_losses)  # type: ignore
        return losses


__all__ = [
    "DDR",
    "DDRLoss",
]
