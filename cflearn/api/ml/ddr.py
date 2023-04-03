import os

import numpy as np

from typing import Any
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from cftool.array import to_numpy
from cftool.array import to_torch
from cftool.types import tensor_dict_type

from ...pipeline import DLInferencePipeline
from ...constants import PREDICTIONS_KEY
from ...misc.toolkit import get_device
from ...misc.toolkit import eval_context
from ...misc.toolkit import show_or_save
from ...models.ml import DDR

try:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure as Figure
except:
    plt = Figure = None


class DDRPredictor:
    def __init__(self, ddr: DDR):
        self.m = ddr
        self.device = get_device(ddr)

    def _fetch(self, x: np.ndarray, **kwargs: Any) -> tensor_dict_type:
        net = to_torch(x).to(self.device)
        with eval_context(self.m, use_grad=True):
            return self.m(net, **kwargs)

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

    @classmethod
    def from_pipeline(cls, m: DLInferencePipeline) -> "DDRPredictor":
        return cls(m.build_model.model)


class DDRVisualizer:
    def __init__(
        self,
        ddr: DDR,
        dpi: int = 200,
        figsize: Tuple[int, int] = (8, 6),
    ):
        if plt is None:
            raise ValueError("`carefree-ml` is needed for `DDRVisualizer`")
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
    ) -> Figure:  # type: ignore
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
            if quantile_curves.shape[-1] != 1:
                raise ValueError("`out_dim` > 1 is not supported")
            quantile_curves = quantile_curves[..., 0]
            for q, quantile_curve in zip(ratios, quantile_curves.T):
                plt.plot(x_base.ravel(), quantile_curve, label=f"quantile {q:4.2f}")
            DDRVisualizer._render_figure(*render_args)
        # cdf curves
        if self.m.predict_cdf:
            anchors = [ratio * (y_max - y_min) + y_min for ratio in ratios]
            cdfs, pdfs = self.predictor.cdf_pdf(x_base, anchors)
            if cdfs.shape[-1] != 1 or pdfs.shape[-1] != 1:
                raise ValueError("`out_dim` > 1 is not supported")
            cdfs, pdfs = cdfs[..., 0], pdfs[..., 0]
            for ratio, anchor, cdf, pdf in zip(ratios, anchors, cdfs.T, pdfs.T):
                anchor_line = np.full(len(x_base), anchor)
                plt.plot(x_base.ravel(), cdf, label=f"cdf {ratio:4.2f}")
                pdf = pdf / (pdf.max() + 1.0e-8) * y_max
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
            yq_predictions = yq_predictions[..., 0]
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
                cdf, pdf = cdf[..., 0], pdf[..., 0]
                cdf = cdf * y_diff + y_min
                _plot("cdf", anchor, yd, cdf, anchor_line)
                pdf = pdf / (pdf.max() + 1.0e-8) * y_max
                _plot("pdf", anchor, None, pdf, anchor_line)

    @classmethod
    def from_pipeline(
        cls,
        m: DLInferencePipeline,
        dpi: int = 200,
        figsize: Tuple[int, int] = (8, 6),
    ) -> "DDRVisualizer":
        return cls(m.build_model.model, dpi, figsize)


__all__ = [
    "DDRPredictor",
    "DDRVisualizer",
]
