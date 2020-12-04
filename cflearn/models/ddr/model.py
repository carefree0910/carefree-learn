import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import Union
from typing import Optional
from cftool.ml import Metrics
from cftool.misc import is_numeric
from cftool.misc import timing_context

from ...misc.toolkit import *
from ..base import ModelBase
from ...types import tensor_dict_type
from ...protocol import TrainerState
from ...modules.transform.core import SplitFeatures


@ModelBase.register("ddr")
@ModelBase.register_pipe("ddr")
class DDR(ModelBase):
    def _init_config(self) -> None:
        super()._init_config()
        # pipe
        head_config = self.get_pipe_config("ddr", "head")
        self.fetch_cdf = head_config.setdefault("fetch_cdf", True)
        self.fetch_q = head_config.setdefault("fetch_q", True)
        if not self.fetch_q and not self.fetch_cdf:
            raise ValueError("something must be fetched, either `q` or `cdf`")
        # common
        self.q_metric = Metrics("quantile")
        self.config.setdefault("ema_decay", 0.0)
        self._synthetic_step = self.config.setdefault("synthetic_step", 10)
        self._synthetic_range = self.config.setdefault("synthetic_range", 3.0)
        labels = self.tr_data.processed.y
        self.y_min, self.y_max = labels.min(), labels.max()
        self.y_diff = self.y_max - self.y_min
        quantile_anchors = np.linspace(0.005, 0.995, 100).astype(np.float32)[..., None]
        y_anchor_choices = quantile_anchors * self.y_diff + self.y_min
        self.register_buffer("quantile_anchors", torch.from_numpy(quantile_anchors))
        self.register_buffer("y_anchor_choices", torch.from_numpy(y_anchor_choices))
        # loss config
        self.config["loss"] = "ddr"
        loss_config = self.config.setdefault("loss_config", {})
        loss_config.setdefault("mtl_method", None)
        loss_config["fetch_q"] = self.fetch_q
        loss_config["fetch_cdf"] = self.fetch_cdf
        # trainer config
        default_metric_types = ["loss"]
        self.quantile_metric_config = None
        if self.fetch_q:
            default_metric_types += ["ddr", "quantile"]
            default_q = np.linspace(0.1, 0.9, 5).tolist()
            qmq = self.config.setdefault("quantile_metric_q", default_q)
            qmq = np.asarray(qmq, np.float32)
            self.quantile_metric_config = {"q": qmq}
        if self.fetch_cdf:
            default_metric_types += ["cdf", "pdf"]
        if self.fetch_q and self.fetch_cdf:
            default_metric_types += ["q_recover", "y_recover"]
        default_metric_weights = {
            "loss": 1.0,
            "ddr": 5.0,
            "quantile": 1.0,
            "cdf": 5.0,
            "pdf": 1.0,
            "q_recover": 10.0,
            "y_recover": 1.0,
        }
        # pipeline config
        # TODO : Support ONNX in DDR
        production = "pipeline"
        # inject new default
        new_default_config = {
            "production": production,
            "trainer_config": {
                "clip_norm": 1.0,
                "num_epoch": 40,
                "max_epoch": 1000,
                "metric_config": {
                    "decay": 0.0,
                    "types": default_metric_types,
                    "weights": default_metric_weights,
                    "quantile_config": self.quantile_metric_config,
                },
            },
        }
        self.environment.update_default_config(new_default_config)

    # utilities

    def _expand(
        self,
        n: int,
        elem: Union[float, torch.Tensor],
        *,
        numpy: bool = False,
        to_device: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(elem, torch.Tensor):
            return elem
        if not is_numeric(elem):
            elem_arr = np.asarray(elem, np.float32)
        else:
            elem_arr = np.repeat(elem, n).astype(np.float32)
        elem_arr = elem_arr.reshape([-1, 1])
        if numpy:
            return elem_arr
        elem_tensor = torch.from_numpy(elem_arr)
        if to_device:
            elem_tensor = elem_tensor.to(self.device)
        return elem_tensor

    # core

    def _quantile(
        self,
        net: Union[torch.Tensor, SplitFeatures],
        q_batch: Optional[torch.Tensor],
        do_inverse: bool,
    ) -> tensor_dict_type:
        kwargs = {
            "ddr": {
                "q_batch": q_batch,
                "median": q_batch is None,
                "do_inverse": do_inverse,
            }
        }
        results = self.execute(net, clear_cache=False, head_kwargs_dict=kwargs)
        ddr_results = results["ddr"]
        return ddr_results  # type: ignore

    def _median(self, net: Union[torch.Tensor, SplitFeatures]) -> tensor_dict_type:
        results = self.execute(
            net,
            clear_cache=False,
            head_kwargs_dict={"ddr": {"median": True}},
        )
        ddr_results = results["ddr"]
        return ddr_results  # type: ignore

    def _cdf(
        self,
        net: Union[torch.Tensor, SplitFeatures],
        y_batch: torch.Tensor,
        need_optimize: bool,
        return_pdf: bool,
        do_inverse: bool,
    ) -> tensor_dict_type:
        use_grad = self.training or return_pdf
        y_batch.requires_grad_(return_pdf)
        with mode_context(self, to_train=None, use_grad=use_grad):
            kwargs = {"ddr": {"y_batch": y_batch, "do_inverse": do_inverse}}
            results = self.execute(net, clear_cache=False, head_kwargs_dict=kwargs)
        ddr_results = results["ddr"]  # type: ignore
        cdf = ddr_results["q"]  # type: ignore
        if not return_pdf:
            pdf = None
        else:
            pdf = get_gradient(cdf, y_batch, need_optimize, need_optimize)
            y_batch.requires_grad_(False)
        return {
            "cdf": cdf,
            "pdf": pdf,
            "cdf_logit": ddr_results["q_logit"],  # type: ignore
            "cdf_logit_mul": ddr_results["q_logit_mul"],  # type: ignore
            "y_inverse_res": ddr_results["y_inverse_res"],  # type: ignore
        }

    def _core(
        self,
        batch_size: int,
        split: SplitFeatures,
        y_batch: torch.Tensor,
        synthetic: bool,
    ) -> tensor_dict_type:
        mask: Optional[torch.Tensor] = None
        if synthetic:
            random_indices = torch.randint(2, [batch_size, 1], device=y_batch.device)
            mask = random_indices == 0
        # TODO : try to optimize this when `fetches` is not full
        # generate quantile / anchor batch
        with timing_context(self, "forward.generate_anchors"):
            q_synthetic_batch = None
            if self.training:
                q_batch = torch.empty_like(y_batch).uniform_()
                y_batch = torch.empty_like(y_batch).uniform_()
                y_batch = y_batch * self.y_diff + self.y_min
                # y_batch = y_batch[np.random.permutation(batch_size)].detach()
                if synthetic:
                    assert mask is not None
                    q_synthetic_batch = torch.empty_like(y_batch)
                    q_synthetic_batch[mask] = 0.25
                    q_synthetic_batch[~mask] = 0.75
            else:
                q_batch = self.quantile_anchors  # type: ignore
                y_batch = self.y_anchor_choices  # type: ignore
                n_repeat = int(batch_size / len(q_batch)) + 1
                q_batch = q_batch.repeat_interleave(n_repeat, dim=0)[:batch_size]
                y_batch = y_batch.repeat_interleave(n_repeat, dim=0)[:batch_size]
        # build predictions
        with timing_context(self, "forward.median"):
            median_rs = {}
            rs = self._median(split)
            if not synthetic:
                median_rs = {
                    "predictions": rs["median"],
                    "med_pos_med_res": rs["pos_med_res"],
                    "med_neg_med_res": rs["neg_med_res"],
                }
        # synthetic predictions
        with timing_context(self, "forward.synthetic"):
            if synthetic:
                if self.fetch_q:
                    assert q_synthetic_batch is not None
                    q_rs = self._quantile(split, q_synthetic_batch, True)
                    median_rs["syn_med_mul"] = q_rs["med_mul"]
                    if not self.fetch_cdf:
                        return median_rs
                if self.fetch_cdf:
                    median = rs["median"].detach()
                    pos_med_res = rs["pos_med_res"]
                    neg_med_res = rs["neg_med_res"]
                    assert mask is not None
                    y_syn_batch = torch.where(mask, pos_med_res, -neg_med_res)
                    y_syn_batch = y_syn_batch.detach() + median
                    y_rs = self._cdf(split, y_syn_batch, False, False, False)
                    y_med_rs = self._cdf(split, median, False, False, False)
                    median_rs.update(
                        {
                            "syn_cdf_logit_mul": y_rs["cdf_logit_mul"],
                            "syn_med_cdf_logit_mul": y_med_rs["cdf_logit_mul"],
                        }
                    )
                    if not self.fetch_q:
                        return median_rs
        # TODO : Some of the calculations in `forward.median` could be reused
        with timing_context(self, "forward.quantile"):
            if not self.fetch_q:
                q_rs = {}
            else:
                q_rs = self._quantile(split, q_batch, True)
        with timing_context(self, "forward.cdf"):
            if not self.fetch_cdf:
                y_rs = {}
            else:
                cdf_inverse = not synthetic
                y_rs = self._cdf(split, y_batch, True, True, cdf_inverse)
        # construct results
        net = list(self._transform_cache.values())[0]
        results: tensor_dict_type = {"net": net, "q_batch": q_batch, "y_batch": y_batch}
        results.update({k: v for k, v in median_rs.items() if v is not None})
        results.update({k: v for k, v in q_rs.items() if v is not None})
        results.update({k: v for k, v in y_rs.items() if v is not None})
        for key in set(median_rs.keys()) | set(q_rs.keys()) | set(y_rs.keys()):
            results.setdefault(key, None)
        return results

    def _predict_quantile(
        self,
        split: SplitFeatures,
        batch_size: int,
        kwargs: Dict[str, Any],
        q: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.fetch_q:
            raise ValueError("quantile function is not fetched")
        if q is not None:
            q = q.tolist()
        else:
            q = kwargs.get("q")
        if q is None:
            raise ValueError(f"quantile cannot be predicted without q")
        quantiles_list, med_mul_list = [], []
        q_list = [q] if isinstance(q, float) else q
        for q in q_list:
            q_batch = self._expand(batch_size, q)
            pack = self._quantile(split, q_batch, False)
            quantiles_list.append(pack["median"] + pack["y_res"])
            med_mul_list.append(pack["med_mul"])
        if len(q_list) == 1:
            return {"quantiles": quantiles_list[0], "med_mul": med_mul_list[0]}
        return {
            "quantiles": torch.cat(quantiles_list, dim=1),
            "med_mul": torch.cat(med_mul_list, dim=1),
        }

    # API

    def forward(
        self,
        batch: tensor_dict_type,
        batch_idx: Optional[int] = None,
        state: Optional[TrainerState] = None,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        # pre-processing
        x_batch = batch["x_batch"]
        y_batch = batch["y_batch"]
        batch_size = x_batch.shape[0]
        split = self._split_features(x_batch, batch_indices, loader_name)
        forward_dict = {}
        # check median residual inference
        predict_mr = kwargs.get("predict_median_residual", False)
        if predict_mr:
            pack = self._median(split)
            median = pack["median"]
            pos, neg = pack["pos_med_res"], pack["neg_med_res"]
            forward_dict["mr_pos"] = pos + median
            forward_dict["mr_neg"] = -neg + median
        # check q inference
        predict_quantile = kwargs.get("predict_quantiles")
        if predict_quantile:
            forward_dict.update(self._predict_quantile(split, batch_size, kwargs))
        # check y inference
        predict_pdf = kwargs.get("predict_pdf", False)
        predict_cdf = kwargs.get("predict_cdf", False)
        if predict_pdf or predict_cdf:
            if not self.fetch_cdf:
                raise ValueError("cdf function is not fetched")
            y = kwargs.get("y")
            if y is None:
                raise ValueError(f"pdf / cdf cannot be predicted without y")
            y_batch = self._expand(batch_size, y, numpy=True)
            y_batch = self.tr_data.transform_labels(y_batch)
            y_batch = to_torch(y_batch).to(self.device)
            forward_dict = self._cdf(split, y_batch, False, predict_pdf, False)
        # check quantile metric
        getting_metrics = kwargs.get("getting_metrics", False)
        if getting_metrics and self.quantile_metric_config is not None:
            q = self.quantile_metric_config["q"]
            forward_dict.update(self._predict_quantile(split, batch_size, kwargs, q))
        if not forward_dict:
            forward_dict = self._core(batch_size, split, y_batch, False)
        self.clear_execute_cache()
        return forward_dict

    def loss_function(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        batch_indices: Optional[torch.Tensor],
        forward_results: tensor_dict_type,
        state: TrainerState,
    ) -> tensor_dict_type:
        y_batch = batch["y_batch"]
        losses, losses_dict = self.loss(forward_results, y_batch)
        net = forward_results["net"]
        if (
            self.training
            and self._synthetic_step > 0
            and state.step % self._synthetic_step == 0
        ):
            with timing_context(self, "synthetic.forward"):
                net_min = torch.min(net, dim=0)[0]
                net_max = torch.max(net, dim=0)[0]
                net_diff = net_max - net_min
                diff_span = 0.5 * (self._synthetic_range - 1.0) * net_diff
                synthetic_net = net.new_empty(net.shape)
                synthetic_net.uniform_(0, 1)
                synthetic_net = self._synthetic_range * synthetic_net * net_diff
                synthetic_net = synthetic_net - (diff_span - net_min)
                # synthetic_net ~ U_[ -diff_span + min, diff_span + max ]
                synthetic_outputs = self._core(
                    net.shape[0],
                    synthetic_net,
                    y_batch,
                    True,
                )
            with timing_context(self, "synthetic.loss"):
                syn_losses, syn_losses_dict = self.loss._core(  # type: ignore
                    synthetic_outputs,
                    y_batch,
                    is_synthetic=True,
                )
            losses_dict.update(syn_losses_dict)
            losses = losses + syn_losses
        losses_dict["loss"] = losses
        losses_dict = {k: v.mean() for k, v in losses_dict.items()}
        if not self.training and self.fetch_q:
            q_losses = []
            y_batch = to_numpy(y_batch)
            for q in np.linspace(0.05, 0.95, 10):
                q = q.item()
                self.q_metric.config["q"] = q
                pack = self._quantile(net, self._expand(len(net), q), False)
                yq = pack["y_res"] + pack["median"]
                q_losses.append(self.q_metric.metric(y_batch, to_numpy(yq)))
            quantile_metric = -sum(q_losses) / len(q_losses) * self.q_metric.sign
            ddr_loss = torch.tensor([quantile_metric], dtype=torch.float32)
            losses_dict["ddr"] = ddr_loss
        self.clear_execute_cache()
        return losses_dict


@ModelBase.register("ddr_q")
class Quantile(DDR):
    def _preset_config(self) -> None:
        head_config = self.get_pipe_config("ddr", "head")
        head_config["fetch_cdf"] = False
        head_config["fetch_q"] = True


@ModelBase.register("ddr_cdf")
class CDF(DDR):
    def _preset_config(self) -> None:
        head_config = self.get_pipe_config("ddr", "head")
        head_config["fetch_cdf"] = True
        head_config["fetch_q"] = False


__all__ = ["CDF", "DDR", "Quantile"]
