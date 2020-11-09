import os

import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from cftool.misc import show_or_save

from ...types import data_type
from ...pipeline.core import Pipeline


class DDRPredictor:
    def __init__(self, ddr: Pipeline):
        self.m = ddr

    def mr(self, x: data_type) -> Dict[str, np.ndarray]:
        return self.m.predict(x, predict_median_residual=True, return_all=True)

    def cdf(self, x: data_type, y: data_type, *, get_pdf: bool = False) -> np.ndarray:
        predictions = self.m.predict(
            x,
            y=y,
            use_grad=get_pdf,
            requires_recover=False,
            predict_pdf=get_pdf,
            predict_cdf=True,
            return_all=True,
        )
        if get_pdf:
            return predictions
        return predictions["cdf"]

    def quantile(
        self,
        x: data_type,
        q: float,
        *,
        recover: bool = True,
    ) -> Dict[str, np.ndarray]:
        return self.m.predict(
            x,
            q=q,
            requires_recover=recover,
            predict_quantiles=True,
            return_all=True,
        )


class DDRVisualizer:
    def __init__(
        self,
        ddr: Pipeline,
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
        median_residual: bool = False,
        add_affine: bool = False,
        mul_affine: bool = False,
        q_batch: Optional[np.ndarray] = None,
        y_batch: Optional[np.ndarray] = None,
        to_pdf: bool = False,
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
        model = self.m.model
        assert model is not None
        affine = add_affine or mul_affine
        mean = None
        median = self.m.predict(x_base)
        fig = self._prepare_base_figure(x, y, x_base, mean, median, indices, "")
        render_args = x_min, x_max, y_min, y_max, y_padding
        # median residual
        if median_residual:
            residuals = self.predictor.mr(x_base)
            pos, neg = map(residuals.get, ["mr_pos", "mr_neg"])
            plt.plot(x_base.ravel(), pos, label="pos_median_residual")
            plt.plot(x_base.ravel(), neg, label="neg_median_residual")
            DDRVisualizer._render_figure(*render_args)
        # quantile curves
        if q_batch is not None and model.fetch_q:
            if not affine:
                for q in q_batch:
                    quantile_curve = self.predictor.quantile(x_base, q)["quantiles"]
                    plt.plot(x_base.ravel(), quantile_curve, label=f"quantile {q:4.2f}")
                DDRVisualizer._render_figure(*render_args)
            else:
                med_add_list, med_mul_list = [], []
                for q in q_batch:
                    med_predictions = self.predictor.quantile(x_base, q, recover=False)
                    med_add_list.append(med_predictions["med_add"])
                    med_mul_list.append(med_predictions["med_mul"])
                if add_affine:
                    for q, med_add in zip(q_batch, med_add_list):
                        plt.plot(x_base.ravel(), med_add, label=f"med_add {q:4.2f}")
                    DDRVisualizer._render_figure(*render_args)
                if mul_affine:
                    for q, med_mul in zip(q_batch, med_mul_list):
                        plt.plot(x_base.ravel(), med_mul, label=f"med_mul {q:4.2f}")
                    DDRVisualizer._render_figure(*render_args)
        # cdf curves
        if y_batch is not None and model.fetch_cdf:
            y_abs_max = np.abs(y).max()
            ratios, anchors = y_batch, [
                ratio * (y_max - y_min) + y_min for ratio in y_batch
            ]
            for ratio, anchor in zip(ratios, anchors):
                anchor_line = np.full(len(x_base), anchor)
                predictions = self.predictor.cdf(x_base, anchor, get_pdf=to_pdf)
                if not to_pdf:
                    cdf = predictions * y_abs_max
                    plt.plot(x_base.ravel(), cdf, label=f"cdf {ratio:4.2f}")
                else:
                    pdf = predictions["pdf"]
                    pdf = pdf * (y_abs_max / max(np.abs(pdf).max(), 1e-8))
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
        x: np.ndarray,
        y: np.ndarray,
        x_base: np.ndarray,
        y_matrix: np.ndarray,
        export_folder: str,
        *,
        q_batch: Optional[np.ndarray] = None,
        y_batch: Optional[np.ndarray] = None,
    ) -> None:
        y_min, y_max = y.min(), y.max()
        y_diff = y_max - y_min

        def _plot(
            prefix: str,
            num: int,
            y_true: Optional[np.ndarray],
            predictions: np.ndarray,
        ) -> None:
            plt.figure(figsize=self.figsize, dpi=self.dpi)
            plt.title(f"{prefix} {num:6.4f}")
            plt.scatter(x[:200], y[:200], color="gray", s=15)
            if y_true is not None:
                plt.plot(x_base, y_true, label="target")
            plt.plot(x_base, predictions, label=f"ddr_prediction")
            plt.legend()
            show_or_save(os.path.join(export_folder, f"{prefix}_{num:4.2f}.png"))

        if q_batch is not None:
            for quantile in q_batch:
                yq = np.percentile(y_matrix, int(100 * quantile), axis=1)
                yq_predictions = self.predictor.quantile(x_base, quantile)["quantiles"]
                _plot("quantile", quantile, yq, yq_predictions)
        if y_batch is not None:
            anchors = [ratio * (y_max - y_min) + y_min for ratio in y_batch]
            for anchor in anchors:
                yd = np.mean(y_matrix <= anchor, axis=1) * y_diff + y_min
                yd_pred = self.predictor.cdf(x_base, anchor)
                yd_pred = yd_pred * y_diff + y_min
                _plot("cdf", anchor, yd, yd_pred)


__all__ = ["DDRPredictor", "DDRVisualizer"]
