import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from cftool.misc import is_numeric
from cftool.misc import show_or_save

from .fcnn import FCNN
from .protocol import MLCoreProtocol
from ..bases import CustomLossBase
from ...types import tensor_dict_type
from ...protocol import losses_type
from ...protocol import LossProtocol
from ...protocol import TrainerState
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY
from ...constants import PREDICTIONS_KEY
from ..implicit.siren import Siren
from ...misc.toolkit import to_numpy
from ...misc.toolkit import to_torch
from ...misc.toolkit import get_gradient


def _expand_tau(
    n: int,
    tau: Union[float, List[float], torch.Tensor],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if isinstance(tau, torch.Tensor):
        return tau
    if not is_numeric(tau):
        tau_arr = np.asarray(tau, np.float32)
    else:
        tau_arr = np.repeat(tau, n).astype(np.float32)
    tau_tensor = torch.from_numpy(tau_arr.reshape([-1, 1]))
    if device is not None:
        tau_tensor = tau_tensor.to(device)
    tau_tensor = tau_tensor * 2.0 - 1.0
    return tau_tensor


@MLCoreProtocol.register("ddr")
class DDR(CustomLossBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_history: int,
        hidden_units: Optional[List[int]] = None,
        *,
        mapping_type: str = "basic",
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
        w_sin: float = 1.0,
        w_sin_initial: float = 30.0,
        num_tau_samples: int = 32,
    ):
        super().__init__(in_dim, out_dim, num_history)
        self.fcnn = FCNN(
            in_dim,
            out_dim,
            num_history,
            hidden_units,
            mapping_type=mapping_type,
            bias=bias,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        hidden_units = self.fcnn.hidden_units
        if not len(set(hidden_units)) == 1:
            raise ValueError("`DDR` requires all hidden units to be identical")
        self.q_siren = Siren(
            None,
            1,
            1,
            hidden_units[0],
            num_layers=len(hidden_units),
            w_sin=w_sin,
            w_sin_initial=w_sin_initial,
            bias=False,
            keep_edge=False,
            use_modulator=False,
        )
        self.num_tau_samples = num_tau_samples

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        # prepare
        net = batch[INPUT_KEY]
        device = net.device
        num_samples = len(net)
        if len(net.shape) > 2:
            net = net.contiguous().view(num_samples, -1)
        # median forward
        mods = []
        for block in self.fcnn.net:
            net = block(net)
            mods.append(net)
        median = mods.pop()
        results = {PREDICTIONS_KEY: median}
        if not kwargs.get("get_quantiles", True):
            return results
        # quantile forward
        tau = kwargs.get("tau", None)
        if tau is not None:
            tau = _expand_tau(num_samples, tau, device)
        else:
            shape = num_samples, self.num_tau_samples, 1
            tau = torch.rand(*shape, device=device) * 2.0 - 1.0
        tau.requires_grad_(True)
        q_increment = self.q_siren(mods, init=tau).squeeze(-1)
        quantiles = median + q_increment
        results.update(
            {
                "tau": tau,
                "q_increment": q_increment,
                "quantiles": quantiles,
            }
        )
        return results

    def _get_losses(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: Any,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> Tuple[tensor_dict_type, tensor_dict_type]:
        state = trainer.state
        forward_results = self(batch_idx, batch, state, **forward_kwargs)
        if not isinstance(trainer.loss, DDRLoss):
            raise ValueError("`DDR` only supports `DDRLoss` as its loss")
        loss_dict = trainer.loss(forward_results, batch, state, **loss_kwargs)
        return forward_results, loss_dict


@LossProtocol.register("ddr")
class DDRLoss(LossProtocol):
    def _init_config(self) -> None:
        self.lb_monotonous = self.config.setdefault("lb_monotonous", 0.1)

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        tau = forward_results["tau"]
        quantiles = forward_results["quantiles"]
        q_increment = forward_results["q_increment"]
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        # mae
        mae = F.l1_loss(predictions, labels, reduction="none")
        # quantiles
        quantile_error = labels - quantiles
        tau_raw = 0.5 * (tau.squeeze(-1).detach() + 1.0)
        neg_errors = tau_raw * quantile_error
        pos_errors = (tau_raw - 1.0) * quantile_error
        q_loss = torch.max(neg_errors, pos_errors).mean(1, keepdim=True)
        # monotonous
        g_tau = get_gradient(q_increment, tau, retain_graph=True, create_graph=True)
        g_tau_loss = F.relu(-g_tau.squeeze(-1), inplace=True).mean(1, keepdim=True)
        # aggregate
        loss = mae + q_loss + self.lb_monotonous * g_tau_loss
        return {"mae": mae, "q_loss": q_loss, "g_tau_loss": g_tau_loss, "loss": loss}


class DDRPredictor:
    def __init__(self, ddr: DDR):
        self.m = ddr

    def _fetch(self, x: np.ndarray, **kwargs) -> tensor_dict_type:
        x_tensor = to_torch(x).to(self.m.device)
        return self.m(0, {INPUT_KEY: x_tensor}, **kwargs)

    def median(self, x: np.ndarray) -> np.ndarray:
        results = self._fetch(x, get_quantiles=False)
        return to_numpy(results[PREDICTIONS_KEY])

    def quantile(self, x: np.ndarray, tau: Union[float, List[float]]) -> np.ndarray:
        results = self._fetch(x, tau=tau)
        return to_numpy(results["quantiles"])


class DDRVisualizer:
    def __init__(
        self,
        ddr: DDR,
        dpi: int = 200,
        figsize: Tuple[int, int] = (8, 6),
    ):
        self.dpi = dpi
        self.figsize = figsize
        self.predictor = DDRPredictor(ddr)

    def _prepare_base_figure(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_base: np.ndarray,
        mean: Optional[np.ndarray],
        median: Optional[np.ndarray],
        indices: np.ndarray,
        title: str,
    ) -> plt.Figure:
        figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.title(title)
        plt.scatter(x[indices], y[indices], color="gray", s=15)
        if mean is not None:
            plt.plot(x_base.ravel(), mean.ravel(), label="mean")
        if median is not None:
            plt.plot(x_base.ravel(), median.ravel(), label="median")
        return figure

    @staticmethod
    def _render_figure(
        x_min: np.ndarray,
        x_max: np.ndarray,
        y_min: float,
        y_max: float,
        y_padding: float,
    ) -> None:
        plt.xlim(x_min, x_max)
        plt.ylim(y_min - 0.5 * y_padding, y_max + 0.5 * y_padding)
        plt.legend()

    def visualize(
        self,
        x: np.ndarray,
        y: np.ndarray,
        export_path: Optional[str],
        *,
        tau: Union[float, List[float]],
        **kwargs: Any,
    ) -> None:
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        padding, dense = kwargs.get("padding", 1.0), kwargs.get("dense", 400)
        indices = np.random.permutation(len(x))[:dense]
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        x_base = np.linspace(x_min, x_max, dense)[..., None]
        mean = None
        median = self.predictor.median(x_base)
        assert isinstance(median, np.ndarray)
        fig = self._prepare_base_figure(x, y, x_base, mean, median, indices, "")
        render_args = x_min, x_max, y_min, y_max, y_padding
        # quantile curves
        quantile_curves = self.predictor.quantile(x_base, tau)
        for q, quantile_curve in zip(tau, quantile_curves.T):
            plt.plot(x_base.ravel(), quantile_curve, label=f"quantile {q:4.2f}")
        DDRVisualizer._render_figure(*render_args)
        show_or_save(export_path, fig)

    def visualize_multiple(
        self,
        f: Callable,
        x: np.ndarray,
        y: np.ndarray,
        export_folder: str,
        *,
        num_base: int = 1000,
        num_repeat: int = 10000,
        tau: Union[float, List[float]],
    ) -> None:
        x_min, x_max = x.min(), x.max()
        x_diff = x_max - x_min
        x_base = np.linspace(x_min - 0.1 * x_diff, x_max + 0.1 * x_diff, num_base)
        x_base = x_base[..., None]
        x_matrix = np.repeat(x_base, num_repeat, axis=1)
        y_matrix = f(x_matrix)
        y_min, y_max = y.min(), y.max()
        y_diff = y_max - y_min

        def _plot(
            prefix: str,
            num: int,
            y_true: Optional[np.ndarray],
            predictions: np.ndarray,
            anchor_line_: Optional[np.ndarray],
        ) -> None:
            plt.figure(figsize=self.figsize, dpi=self.dpi)
            plt.title(f"{prefix} {num:6.4f}")
            plt.scatter(x[:200], y[:200], color="gray", s=15)
            if y_true is not None:
                plt.plot(x_base, y_true, label="target")
            plt.plot(x_base, predictions, label=f"ddr_prediction")
            if anchor_line_ is not None:
                plt.plot(x_base.ravel(), anchor_line_, color="gray")
            plt.legend()
            show_or_save(os.path.join(export_folder, f"{prefix}_{num:4.2f}.png"))

        yq_predictions = self.predictor.quantile(x_base, tau)
        for q, yq_pred in zip(tau, yq_predictions.T):
            yq = np.percentile(y_matrix, int(100 * q), axis=1)
            _plot("quantile", q, yq, yq_pred, None)


__all__ = [
    "DDR",
    "DDRLoss",
    "DDRPredictor",
    "DDRVisualizer",
]
