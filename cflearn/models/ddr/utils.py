import os

import numpy as np
import matplotlib.pyplot as plt

from cftool.misc import show_or_save


class DDRPredictor:
    def __init__(self, ddr):
        self.m = ddr

    def mr(self, x):
        return self.m.predict(x, predict_median_residual=True, return_all=True)

    def cdf(self, x, y, *, get_pdf: bool = False):
        predictions = self.m.predict(
            x,
            y=y,
            no_grad=not get_pdf,
            requires_recover=False,
            predict_pdf=get_pdf,
            predict_cdf=True,
            return_all=True,
        )
        if get_pdf:
            return predictions
        return predictions["cdf"]

    def quantile(self, x, q):
        predictions = self.m.predict(x, q=q, predict_quantile=True, return_all=True)
        return predictions["quantile"]


class DDRVisualizer:
    def __init__(self, ddr):
        self.m = ddr
        self.predictor = DDRPredictor(ddr)

    @staticmethod
    def _prepare_base_figure(x, y, x_base, mean, median, indices, title):
        figure = plt.figure()
        plt.title(title)
        plt.scatter(x[indices], y[indices], color="gray", s=15)
        if mean is not None:
            plt.plot(x_base.ravel(), mean.ravel(), label="mean")
        plt.plot(x_base.ravel(), median.ravel(), label="median")
        return figure

    @staticmethod
    def _render_figure(x_min, x_max, y_min, y_max, y_padding):
        plt.xlim(x_min, x_max)
        plt.ylim(y_min - 0.5 * y_padding, y_max + 0.5 * y_padding)
        plt.legend()

    def visualize(
        self,
        x,
        y,
        export_path,
        *,
        residual=False,
        quantiles=None,
        anchor_ratios=None,
        **kwargs,
    ):
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
        median = self.m.predict(x_base)
        fig = DDRVisualizer._prepare_base_figure(
            x, y, x_base, mean, median, indices, ""
        )
        render_args = x_min, x_max, y_min, y_max, y_padding
        # median residual
        if residual:
            residuals = self.predictor.mr(x_base)
            pos, neg = map(residuals.get, ["mr_pos", "mr_neg"])
            plt.plot(x_base.ravel(), pos, label="pos_median_residual")
            plt.plot(x_base.ravel(), neg, label="neg_median_residual")
            DDRVisualizer._render_figure(*render_args)
        # quantile curves
        if quantiles is not None:
            for q in quantiles:
                quantile_curve = self.predictor.quantile(x_base, q)
                plt.plot(x_base.ravel(), quantile_curve, label=f"quantile {q:4.2f}")
            DDRVisualizer._render_figure(*render_args)
        # cdf curves
        if anchor_ratios is not None:
            y_abs_max = np.abs(y).max()
            ratios, anchors = anchor_ratios, [
                ratio * (y_max - y_min) + y_min for ratio in anchor_ratios
            ]
            for ratio, anchor in zip(ratios, anchors):
                predictions = self.predictor.cdf(x_base, anchor, get_pdf=True)
                pdf, cdf = map(predictions.get, ["pdf", "cdf"])
                pdf, cdf = (
                    pdf * (y_abs_max / max(np.abs(pdf).max(), 1e-8)),
                    cdf * y_abs_max,
                )
                anchor_line = np.full(len(x_base), anchor)
                plt.plot(x_base.ravel(), pdf, label=f"pdf {ratio:4.2f}")
                plt.plot(x_base.ravel(), cdf, label=f"cdf {ratio:4.2f}")
                plt.plot(
                    x_base.ravel(),
                    anchor_line,
                    label=f"anchor {ratio:4.2f}",
                    color="gray",
                )
            DDRVisualizer._render_figure(x_min, x_max, y_min, y_max, y_padding)
        show_or_save(export_path, fig)

    def visualize_multiple(
        self,
        x,
        y,
        x_base,
        y_matrix,
        export_folder,
        *,
        quantiles=None,
        anchor_ratios=None,
    ):
        y_min, y_max = y.min(), y.max()
        y_diff = y_max - y_min

        def _plot(prefix, num, y_true, predictions, dual_predictions):
            def _core(pred, *, dual):
                suffix = "_dual" if dual else ""
                plt.figure()
                plt.title(f"{prefix} {num:6.4f}")
                plt.scatter(x[:200], y[:200], color="gray", s=15)
                plt.plot(x_base, y_true, label="target")
                plt.plot(x_base, pred, label=f"ddr{suffix}_prediction")
                plt.legend()
                show_or_save(
                    os.path.join(export_folder, f"{prefix}_{num:4.2f}{suffix}.png")
                )

            _core(predictions, dual=False)
            if dual_predictions is not None:
                _core(dual_predictions, dual=True)

        if quantiles is not None:
            for quantile in quantiles:
                yq = np.percentile(y_matrix, int(100 * quantile), axis=1)
                yq_pred = self.predictor.quantile(x_base, quantile)
                # yqd_pred = ddr.quantile(quantile, x_base, inference_only=True)
                yqd_pred = None
                _plot("quantile", quantile, yq, yq_pred, yqd_pred)
            plt.figure()
            for quantile in [0.25, 0.5, 0.75]:
                yq = np.percentile(y_matrix, int(100 * quantile), axis=1)
                plt.title(f"quantile {quantile:4.2f}")
                plt.scatter(x[:200], y[:200], color="gray", s=15)
                plt.plot(x_base, yq, label=f"{quantile:3.2f}")
                plt.legend()
            export_path = os.path.join(export_folder, "median_residual.png")
            self.visualize(x, y, export_path, residual=True)
        if anchor_ratios is not None:
            anchors = [ratio * (y_max - y_min) + y_min for ratio in anchor_ratios]
            for anchor in anchors:
                yd = np.mean(y_matrix <= anchor, axis=1) * y_diff + y_min
                yd_pred = self.predictor.cdf(x_base, anchor)
                yd_pred = yd_pred * y_diff + y_min
                # ydq_pred = ddr.cdf(data_anchor, x_base, inference_only=True) * y_diff + y_min
                ydq_pred = None
                _plot("cdf", anchor, yd, yd_pred, ydq_pred)


__all__ = ["DDRPredictor", "DDRVisualizer"]
