import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from cftool.misc import show_or_save

from .fcnn import FCNN
from .protocol import MLCoreProtocol
from ..bases import CustomLossBase
from ...types import tensor_dict_type
from ...protocol import losses_type
from ...protocol import LossProtocol
from ...protocol import TrainerState
from ...constants import LOSS_KEY
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY
from ...constants import PREDICTIONS_KEY
from ..implicit.siren import _make_grid
from ..implicit.siren import Siren
from ...misc.toolkit import to_numpy
from ...misc.toolkit import to_torch
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
    return _make_grid(num_samples + 2, 1, device)[:, 1:-1]


@MLCoreProtocol.register("ddr")
class DDR(CustomLossBase):
    use_grad_in_summary = True

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
        num_random_samples: int = 16,
        predict_quantiles: bool = True,
        predict_cdf: bool = True,
        y_min_max: Optional[Tuple[float, float]] = None,
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
        assert hidden_units is not None
        if not len(set(hidden_units)) == 1:
            raise ValueError("`DDR` requires all hidden units to be identical")

        def _make_siren(_in_dim: Optional[int] = None) -> Siren:
            return Siren(
                None,
                _in_dim or 1,
                out_dim,
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
        self.cdf_siren = None if not predict_cdf else _make_siren(out_dim)
        self.num_random_samples = num_random_samples
        self._y_min_max = y_min_max

    def _init_with_trainer(self, trainer: Any) -> None:
        if self._y_min_max is None:
            y_train = trainer.train_loader.data.y
            self._y_min_max = y_train.min().item(), y_train.max().item()
        self.register_buffer("y_min_max", torch.tensor(self._y_min_max))

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
        logit = self.cdf_siren(mods, init=y_ratio).squeeze(-1)  # type: ignore
        cdf = torch.sigmoid(logit)
        return y_ratio, logit, cdf

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
        get_quantiles = kwargs.get("get_quantiles", True) and self.predict_quantiles
        get_cdf = kwargs.get("get_cdf", True) and self.predict_cdf
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
            tau = kwargs.get("tau", None)
            if tau is not None:
                tau = _expand_element(num_samples, tau, device) * 2.0 - 1.0
            else:
                shape = num_samples, self.num_random_samples, 1
                if self.training:
                    tau = torch.rand(*shape, device=device) * 2.0 - 1.0
                else:
                    tau = _make_ddr_grid(self.num_random_samples, device)
                    tau = tau.repeat(num_samples, 1, 1)
            tau.requires_grad_(True)
            q_increment, quantiles = self._get_quantiles(tau, mods, median)
            results.update(
                {
                    "tau": tau,
                    "q_increment": q_increment,
                    "quantiles": quantiles,
                }
            )
        # cdf forward
        if get_cdf:
            y_anchor = kwargs.get("y_anchor", None)
            if y_anchor is not None:
                y_anchor = _expand_element(num_samples, y_anchor, device)
            else:
                shape = num_samples, self.num_random_samples, 1
                if self.training:
                    y_anchor = torch.rand(*shape, device=device) * y_span + y_min
                else:
                    y_raw_ratio = _make_ddr_grid(self.num_random_samples, device)
                    y_raw_ratio = 0.5 * (y_raw_ratio + 1.0)
                    y_anchor = (y_raw_ratio * y_span + y_min).repeat(num_samples, 1, 1)
            y_anchor.requires_grad_(True)
            y_ratio, logit, cdf = self._get_cdf(y_anchor, median, y_span, mods)
            pdf = get_gradient(cdf, y_anchor, True, True).squeeze(-1)  # type: ignore
            results.update(
                {
                    "y_anchor": y_anchor.squeeze(-1),
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
        self.lb_ddr = self.config.setdefault("lb_ddr", 1.0)
        self.lb_dual = self.config.setdefault("lb_dual", 1.0)
        self.lb_monotonous = self.config.setdefault("lb_monotonous", 1.0)

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
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
        # TODO : double check the following implementations
        # quantiles
        if all_exists(tau, quantiles, q_increment):
            quantile_error = labels[:, None] - quantiles
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
            tau_raw = 0.5 * (tau.squeeze(-1).detach() + 1.0)  # type: ignore
            tau_recover_loss = F.l1_loss(tau_raw, dual_cdf)  # type: ignore
            losses["tau_recover"] = tau_recover_loss
            weighted_losses.append(self.lb_dual * tau_recover_loss)
        # aggregate
        losses[LOSS_KEY] = sum(weighted_losses)  # type: ignore
        return losses


class DDRPredictor:
    def __init__(self, ddr: DDR):
        self.m = ddr

    def _fetch(self, x: np.ndarray, **kwargs: Any) -> tensor_dict_type:
        x_tensor = to_torch(x).to(self.m.device)
        return self.m(0, {INPUT_KEY: x_tensor}, **kwargs)

    def median(self, x: np.ndarray) -> np.ndarray:
        results = self._fetch(x, get_quantiles=False)
        return to_numpy(results[PREDICTIONS_KEY])

    def quantile(self, x: np.ndarray, tau: Union[float, List[float]]) -> np.ndarray:
        results = self._fetch(x, tau=tau, get_cdf=False)
        return to_numpy(results["quantiles"])

    def cdf_pdf(
        self,
        x: np.ndarray,
        y: Union[float, List[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(y, float):
            y = [y]
        y_anchor = np.array(y, np.float32).reshape([1, -1])
        y_anchor = np.repeat(y_anchor, len(x), axis=0)
        results = self._fetch(x, y_anchor=y_anchor, get_quantiles=False)
        return to_numpy(results["cdf"]), to_numpy(results["pdf"])


class DDRVisualizer:
    def __init__(
        self,
        ddr: DDR,
        dpi: int = 200,
        figsize: Tuple[int, int] = (8, 6),
    ):
        self.m = ddr
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
        ratios: Union[float, List[float]],
        **kwargs: Any,
    ) -> None:
        if isinstance(ratios, float):
            ratios = [ratios]

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
        if self.m.predict_quantiles:
            quantile_curves = self.predictor.quantile(x_base, ratios)
            for q, quantile_curve in zip(ratios, quantile_curves.T):
                plt.plot(x_base.ravel(), quantile_curve, label=f"quantile {q:4.2f}")
            DDRVisualizer._render_figure(*render_args)
        # cdf curves
        if self.m.predict_cdf:
            anchors = [ratio * (y_max - y_min) + y_min for ratio in ratios]
            cdfs, pdfs = self.predictor.cdf_pdf(x_base, anchors)
            for ratio, anchor, cdf, pdf in zip(ratios, anchors, cdfs.T, pdfs.T):
                anchor_line = np.full(len(x_base), anchor)
                plt.plot(x_base.ravel(), cdf, label=f"cdf {ratio:4.2f}")
                plt.plot(x_base.ravel(), pdf, label=f"pdf {ratio:4.2f}")
                plt.plot(
                    x_base.ravel(),
                    anchor_line,
                    label=f"anchor {ratio:4.2f}",
                    color="gray",
                )
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
        ratios: Union[float, List[float]],
    ) -> None:
        if isinstance(ratios, float):
            ratios = [ratios]

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
            num: float,
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

        if self.m.predict_quantiles:
            yq_predictions = self.predictor.quantile(x_base, ratios)
            for q, yq_pred in zip(ratios, yq_predictions.T):
                yq = np.percentile(y_matrix, int(100 * q), axis=1)
                _plot("quantile", q, yq, yq_pred, None)

        if self.m.predict_cdf:
            anchors = [ratio * (y_max - y_min) + y_min for ratio in ratios]
            for anchor in anchors:
                assert isinstance(anchor, float)
                anchor_line = np.full(len(x_base), anchor)
                yd = np.mean(y_matrix <= anchor, axis=1) * y_diff + y_min
                cdf, pdf = self.predictor.cdf_pdf(x_base, anchor)
                cdf = cdf * y_diff + y_min
                _plot("cdf", anchor, yd, cdf, anchor_line)
                _plot("pdf", anchor, None, pdf, anchor_line)


__all__ = [
    "DDR",
    "DDRLoss",
    "DDRPredictor",
    "DDRVisualizer",
]
