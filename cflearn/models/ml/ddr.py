import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from cftool.types import tensor_dict_type

from .base import MLModel
from .fcnn import FCNN
from ...types import losses_type
from ...schema import ILoss
from ...schema import ITrainer
from ...schema import TrainerState
from ...schema import MLEncoderSettings
from ...schema import MLGlobalEncoderSettings
from ...constants import LOSS_KEY
from ...constants import LABEL_KEY
from ...constants import PREDICTIONS_KEY
from ..implicit.siren import make_grid
from ..implicit.siren import Siren
from ...misc.toolkit import get_gradient


def all_exists(*tensors: Optional[Tensor]) -> bool:
    return all(tensor is not None for tensor in tensors)


def _expand_element(
    n: int,
    element: Union[float, List[float], np.ndarray, Tensor],
    device: Optional[torch.device] = None,
) -> Tensor:
    if isinstance(element, Tensor):
        return element
    if isinstance(element, np.ndarray):
        element_arr = element.reshape(*element.shape, 1)
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
    return make_grid(num_samples + 2, 1, device)[:, 1:-1]


@MLModel.register("ddr")
class DDR(MLModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int = 1,
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
        predict_quantiles: bool = True,
        predict_cdf: bool = True,
        y_min_max: Optional[Tuple[float, float]] = None,
        encoder_settings: Optional[Dict[str, MLEncoderSettings]] = None,
        global_encoder_settings: Optional[MLGlobalEncoderSettings] = None,
    ):
        super().__init__(
            encoder_settings=encoder_settings,
            global_encoder_settings=global_encoder_settings,
        )
        if self.encoder is not None:
            input_dim += self.encoder.dim_increment
        self.fcnn = FCNN(
            input_dim,
            output_dim,
            num_history,
            hidden_units,
            mapping_type=mapping_type,
            bias=bias,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        hidden_units = self.fcnn.hidden_units
        assert hidden_units is not None
        if not len(set(hidden_units)) == 1:
            raise ValueError("`DDR` requires all hidden units to be identical")

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
                keep_edge=False,
                use_modulator=False,
            )

        self.predict_quantiles = predict_quantiles
        self.predict_cdf = predict_cdf
        self.q_siren = None if not predict_quantiles else _make_siren()
        self.cdf_siren = None if not predict_cdf else _make_siren(output_dim)
        self.num_random_samples = num_random_samples
        self._y_min_max = y_min_max
        self.register_buffer("y_min_max", torch.tensor([0.0, 0.0]))

    # inheritance

    def forward(
        self,
        net: Tensor,
        *,
        get_cdf: bool = True,
        get_quantiles: bool = True,
        tau: Optional[float] = None,
        y_anchor: Optional[float] = None,
    ) -> tensor_dict_type:
        # prepare
        device = net.device
        num_samples = len(net)
        if len(net.shape) > 2:
            net = net.contiguous().view(num_samples, -1)
        get_quantiles = get_quantiles and self.predict_quantiles
        get_cdf = get_cdf and self.predict_cdf
        # median forward
        mods = []
        for block in self.fcnn.net:
            net = block(net)
            mods.append(net)
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
                shape = num_samples, self.num_random_samples, 1
                if self.training:
                    tau_tensor = torch.rand(*shape, device=device) * 2.0 - 1.0
                else:
                    tau_tensor = _make_ddr_grid(self.num_random_samples, device)
                    tau_tensor = tau_tensor.repeat(num_samples, 1, 1)
            tau_tensor.requires_grad_(True)
            q_increment, quantiles = self._get_quantiles(tau_tensor, mods, median)
            results.update(
                {
                    "tau": tau_tensor,
                    "q_increment": q_increment,
                    "quantiles": quantiles,
                }
            )
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
            y_ratio, logit, cdf = self._get_cdf(ya_tensor, median, y_span, mods)
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
        if get_quantiles and get_cdf:
            dual_y = results["quantiles"].detach()
            results["dual_cdf"] = self._get_cdf(dual_y, median, y_span, mods)[-1]
        return results

    def init_with_trainer(self, trainer: ITrainer) -> None:
        if self._y_min_max is None:
            y_train = trainer.train_loader.get_full_batch()[LABEL_KEY]
            self._y_min_max = y_train.min().item(), y_train.max().item()
        self.y_min_max = torch.tensor(self._y_min_max)

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
        y_residual = y_anchor - median.unsqueeze(1)
        y_ratio = y_residual / y_span
        logit = self.cdf_siren(mods, init=y_ratio)  # type: ignore
        cdf = torch.sigmoid(logit)
        return y_ratio, logit, cdf


@ILoss.register("ddr")
class DDRLoss(ILoss):
    def __init__(
        self,
        *,
        lb_ddr: float = 1.0,
        lb_dual: float = 1.0,
        lb_monotonous: float = 1.0,
    ):
        super().__init__()
        self.lb_ddr = lb_ddr
        self.lb_dual = lb_dual
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
        median = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        losses = {}
        weighted_losses = []
        # mae
        mae = F.l1_loss(median, labels, reduction="none")
        losses["mae"] = mae
        weighted_losses.append(mae)
        # quantiles
        labels = labels[:, None]
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
            cdf_loss = -indicative * logit + F.softplus(logit)
            cdf_loss = cdf_loss.mean((1, 2), keepdim=True)
            pdf_loss = F.relu(-pdf, inplace=True).mean((1, 2), keepdim=True)  # type: ignore
            losses["cdf"] = cdf_loss
            losses["pdf"] = pdf_loss
            weighted_losses.append(self.lb_ddr * cdf_loss)
            weighted_losses.append(self.lb_monotonous * pdf_loss)
        # dual
        if all_exists(dual_cdf):
            tau_raw = 0.5 * (tau.detach() + 1.0)  # type: ignore
            tau_recover_loss = F.l1_loss(tau_raw, dual_cdf)  # type: ignore
            losses["tau_recover"] = tau_recover_loss
            weighted_losses.append(self.lb_dual * tau_recover_loss)
        # aggregate
        losses[LOSS_KEY] = sum(weighted_losses)  # type: ignore
        return losses


__all__ = [
    "DDR",
    "DDRLoss",
]
