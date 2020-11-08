import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import Union
from typing import Optional
from cftool.ml import Metrics
from cftool.misc import is_numeric
from cftool.misc import update_dict
from cftool.misc import timing_context
from cfdata.tabular import DataLoader

from ...misc.toolkit import *
from .core import DDRCore
from .loss import DDRLoss
from ..base import ModelBase
from ...types import tensor_dict_type


@ModelBase.register("ddr")
class DDR(ModelBase):
    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        tr_loader: DataLoader,
        cv_loader: DataLoader,
        tr_weights: Optional[np.ndarray],
        cv_weights: Optional[np.ndarray],
        device: torch.device,
        *,
        use_tqdm: bool,
    ):
        if not tr_loader.data.task_type.is_reg:
            raise ValueError("DDR can only deal with regression problems")
        super().__init__(
            pipeline_config,
            tr_loader,
            cv_loader,
            tr_weights,
            cv_weights,
            device,
            use_tqdm=use_tqdm,
        )
        self.q_metric = Metrics("quantile")
        cfg = self.get_core_config(self)
        assert cfg.pop("out_dim") == 1
        self.core = DDRCore(**cfg)

    @property
    def input_sample(self) -> tensor_dict_type:
        return super().input_sample

    @staticmethod
    def get_core_config(instance: "ModelBase") -> Dict[str, Any]:
        cfg = ModelBase.get_core_config(instance)
        num_layers = instance.config.setdefault("num_layers", None)
        num_blocks = instance.config.setdefault("num_blocks", None)
        latent_dim = instance.config.setdefault("latent_dim", None)
        transition_builder = instance.config.setdefault("transition_builder", None)
        cfg.update(
            {
                "y_min": instance.y_min,
                "y_max": instance.y_max,
                "fetch_q": instance.fetch_q,
                "fetch_cdf": instance.fetch_cdf,
                "num_layers": num_layers,
                "num_blocks": num_blocks,
                "latent_dim": latent_dim,
                "transition_builder": transition_builder,
            }
        )
        return cfg

    @property
    def fetch_q(self) -> bool:
        return "q" in self.fetches

    @property
    def fetch_cdf(self) -> bool:
        return "cdf" in self.fetches

    def _init_config(self) -> None:
        super()._init_config()
        # common
        self.config.setdefault("ema_decay", 0.0)
        self.fetches = set(self.config.setdefault("fetches", {"q", "cdf"}))
        if not self.fetch_q and not self.fetch_cdf:
            raise ValueError("something must be fetched, either `q` or `cdf`")
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
        self._loss_config = self.config.setdefault("loss_config", {})
        self._loss_config.setdefault("mtl_method", None)
        self._loss_config["fetch_q"] = self.fetch_q
        self._loss_config["fetch_cdf"] = self.fetch_cdf
        # trainer config
        default_metric_types = []
        if self.fetch_q:
            default_metric_types += ["ddr"]
        if self.fetch_cdf:
            default_metric_types += ["cdf", "pdf"]
        if self.fetch_q and self.fetch_cdf:
            default_metric_types += ["q_recover", "y_recover", "loss"]
        default_metric_weights = {
            "ddr": 5.0,
            "cdf": 1.0,
            "pdf": 1.0,
            "loss": 1.0,
        }
        trainer_config = self._pipeline_config.setdefault("trainer_config", {})
        trainer_config = update_dict(
            trainer_config,
            {
                "clip_norm": 1.0,
                "num_epoch": 40,
                "max_epoch": 1000,
                "metric_config": {
                    "decay": 0.0,
                    "types": default_metric_types,
                    "weights": default_metric_weights,
                },
            },
        )
        self._pipeline_config["trainer_config"] = trainer_config
        self._trainer_config = trainer_config

    def _init_loss(self) -> None:
        self.loss = DDRLoss(self._loss_config, "none")

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
        net: torch.Tensor,
        q_batch: Optional[torch.Tensor],
        do_inverse: bool,
        return_mr: bool = False,
    ) -> tensor_dict_type:
        return self.core(
            net,
            q_batch=q_batch,
            median=q_batch is None,
            return_mr=return_mr,
            do_inverse=do_inverse,
        )

    def _median(
        self,
        net: torch.Tensor,
        do_inverse: bool,
        return_mr: bool,
    ) -> tensor_dict_type:
        return self._quantile(net, None, do_inverse, return_mr)

    def _cdf(
        self,
        net: torch.Tensor,
        y_batch: torch.Tensor,
        need_optimize: bool,
        return_pdf: bool,
        do_inverse: bool,
    ) -> tensor_dict_type:
        use_grad = self.training or return_pdf
        y_batch.requires_grad_(return_pdf)
        with mode_context(self, to_train=None, use_grad=use_grad):
            results = self.core(net, y_batch=y_batch, do_inverse=do_inverse)
        if not return_pdf:
            pdf = None
        else:
            cdf = results["q"]
            pdf = get_gradient(cdf, y_batch, need_optimize, need_optimize)
            assert isinstance(pdf, torch.Tensor)
            y_batch.requires_grad_(False)
        results["pdf"] = pdf
        return results

    def _core(
        self,
        net: torch.Tensor,
        y_batch: torch.Tensor,
        synthetic: bool,
    ) -> tensor_dict_type:
        batch_size = len(net)
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
                    q_synthetic_batch = torch.empty_like(y_batch)
                    random_indices = torch.randint(2, [batch_size])
                    mask = random_indices == 0
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
            if self.fetch_q:
                rs = self._quantile(net, None, False)
                median_rs = {
                    "median_med_add": rs["med_add"],
                    "median_med_mul": rs["med_mul"],
                }
                if synthetic:
                    assert q_synthetic_batch is not None
                    rs = self._quantile(net, q_synthetic_batch, False, True)
                    median_rs.update(
                        {
                            "syn_med_add": rs["med_add"],
                            "syn_med_mul": rs["med_mul"],
                            "syn_med_res": rs["med_res"],
                        }
                    )
                    if not self.fetch_cdf:
                        return median_rs
                else:
                    median_rs["predictions"] = rs["median"]
                    if self.fetch_cdf:
                        median_rs["median_inverse"] = rs["q_inverse"]
        # TODO : Some of the calculations in `forward.median` could be reused
        with timing_context(self, "forward.quantile"):
            if not self.fetch_q:
                q_rs = {}
            else:
                q_rs = self._quantile(net, q_batch, True)
        with timing_context(self, "forward.cdf"):
            if not self.fetch_cdf:
                y_rs = {}
            else:
                cdf_inverse = not synthetic
                y_rs = self._cdf(net, y_batch, True, True, cdf_inverse)
                q = y_rs["cdf"] = y_rs.pop("q")
                y_rs["cdf_logit"] = y_rs.pop("q_logit")
                if not self.fetch_q:
                    y_rs["predictions"] = q
        # construct results
        results: tensor_dict_type = {"net": net, "q_batch": q_batch, "y_batch": y_batch}
        results.update({k: v for k, v in median_rs.items() if v is not None})
        results.update({k: v for k, v in q_rs.items() if v is not None})
        results.update({k: v for k, v in y_rs.items() if v is not None})
        for key in set(median_rs.keys()) | set(q_rs.keys()) | set(y_rs.keys()):
            results.setdefault(key, None)
        return results

    # API

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        batch_step: int = 0,
        **kwargs: Any,
    ) -> tensor_dict_type:
        # pre-processing
        x_batch = batch["x_batch"]
        batch_size = x_batch.shape[0]
        net = self._split_features(x_batch, batch_indices, loader_name).merge()
        if self.tr_data.is_ts:
            net = net.view(batch_size, -1)
        y_batch = batch["y_batch"]
        # check q inference
        forward_dict = {}
        predict_mr = kwargs.get("predict_median_residual", False)
        predict_quantile = kwargs.get("predict_quantiles")
        if predict_mr or predict_quantile:
            if not self.fetch_q:
                raise ValueError("quantile function is not fetched")
            if predict_mr:
                pack = self._median(net, False, True)
                median = pack["median"]
                pos, neg = pack["pos_med_res"], pack["neg_med_res"]
                forward_dict["mr_pos"] = pos + median
                forward_dict["mr_neg"] = -neg + median
            if predict_quantile:
                q = kwargs.get("q")
                if q is None:
                    raise ValueError(f"quantile cannot be predicted without q")
                q_batch = self._expand(batch_size, q)
                pack = self._quantile(net, q_batch, False)
                forward_dict["quantiles"] = pack["median"] + pack["y_res"]
                forward_dict["med_add"] = pack["med_add"]
                forward_dict["med_mul"] = pack["med_mul"]
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
            results = self._cdf(net, y_batch, False, predict_pdf, False)
            if predict_pdf:
                forward_dict["pdf"] = results["pdf"]
            if predict_cdf:
                forward_dict["cdf"] = results["q"]
        if not forward_dict:
            forward_dict = self._core(net, y_batch, False)
        return forward_dict

    def loss_function(
        self,
        batch: tensor_dict_type,
        batch_indices: np.ndarray,
        forward_results: tensor_dict_type,
        batch_step: int,
    ) -> tensor_dict_type:
        y_batch = batch["y_batch"]
        losses, losses_dict = self.loss(forward_results, y_batch)
        net = forward_results["net"]
        if (
            self.training
            and self._synthetic_step > 0
            and batch_step % self._synthetic_step == 0
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
                synthetic_outputs = self._core(synthetic_net, y_batch, True)
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
            assert isinstance(y_batch, torch.Tensor)
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
        return losses_dict


__all__ = ["DDR"]
