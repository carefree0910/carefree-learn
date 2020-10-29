import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from cftool.ml import Metrics
from cftool.misc import is_numeric
from cftool.misc import update_dict
from cftool.misc import timing_context
from cftool.misc import shallow_copy_dict
from cfdata.tabular import DataLoader
from ...types import tensor_dict_type

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None

from ...misc.toolkit import *
from ...modules.blocks import *
from .core import DDRCore
from .loss import DDRLoss
from ..base import ModelBase

proj_type = Optional[Union[Linear, MLP]]
tensors_type = Union[
    torch.Tensor,
    List[torch.Tensor],
    Dict[str, Optional[Union[torch.Tensor, tensor_dict_type]]],
]


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
        num_blocks = instance.config.setdefault("num_blocks", 3)
        to_latent = instance.config.setdefault("to_latent", True)
        latent_dim = instance.config.setdefault("latent_dim", 256)
        num_units = instance.config.setdefault("num_units", None)
        mapping_configs = instance.config.setdefault("mapping_configs", None)
        cfg.update(
            {
                "num_blocks": num_blocks,
                "to_latent": to_latent,
                "latent_dim": latent_dim,
                "num_units": num_units,
                "mapping_configs": mapping_configs,
            }
        )
        return cfg

    @property
    def default_anchors(self) -> np.ndarray:
        return np.linspace(0.05, 0.95, 10).astype(np.float32)

    def _init_config(self) -> None:
        super()._init_config()
        # common
        self._step_count = 0
        self._synthetic_step = int(self.config.setdefault("synthetic_step", 5))
        self._use_gradient_loss = self.config.setdefault("use_gradient_loss", True)
        y_anchors = self.config.setdefault("cdf_ratio_anchors", self.default_anchors)
        q_anchors = self.config.setdefault("quantile_anchors", self.default_anchors)
        labels = self.tr_data.processed.y
        y_min, y_max = labels.min(), labels.max()
        self.y_min = y_min
        self.y_diff = y_max - y_min
        if y_anchors is None:
            self._anchor_choice_array = None
        else:
            anchors = np.asarray(y_anchors, np.float32)
            self._anchor_choice_array = y_min + (y_max - y_min) * anchors
        if q_anchors is None:
            self._quantile_anchors = None
        else:
            self._quantile_anchors = np.asarray(q_anchors, np.float32)
        self._synthetic_range = self.config.setdefault("synthetic_range", 3)
        # loss config
        self._loss_config = self.config.setdefault("loss_config", {})
        self._loss_config.setdefault("use_anneal", True)
        self._loss_config.setdefault("mtl_method", None)
        # trainer config
        default_metric_types = ["ddr", "loss"]
        trainer_config = self._pipeline_config.setdefault("trainer_config", {})
        trainer_config = update_dict(
            trainer_config,
            {
                "clip_norm": 1.0,
                "num_epoch": 40,
                "max_epoch": 1000,
                "metric_config": {"types": default_metric_types, "decay": 0.5},
            },
        )
        num_epoch = trainer_config["num_epoch"]
        num_train_samples = self.tr_data.processed.x.shape[0]
        batch_size = self._pipeline_config.setdefault("batch_size", 128)
        anneal_step = self._loss_config.setdefault(
            "anneal_step", (num_train_samples * num_epoch) // (batch_size * 2)
        )
        self._loss_config.setdefault("anneal_step", anneal_step)
        self._pipeline_config["trainer_config"] = trainer_config
        self._trainer_config = trainer_config

    def _init_loss(self) -> None:
        loss_config = shallow_copy_dict(self._loss_config)
        self.loss = DDRLoss(loss_config, "none")

    # utilities

    def _convert_np_anchors(self, np_anchors: np.ndarray) -> torch.Tensor:
        tensor = to_torch(np_anchors.reshape([-1, 1]))
        return tensor.to(self.device).requires_grad_(True)

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

    def _sample_anchors(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        choice_indices = np.random.randint(0, len(self._quantile_anchors), n)
        chosen = self._quantile_anchors[choice_indices]
        sampled_q_batch = self._convert_np_anchors(chosen)
        choice_indices = np.random.randint(0, len(self._anchor_choice_array), n)
        chosen = self._anchor_choice_array[choice_indices]
        sampled_y_batch = self._convert_np_anchors(chosen)
        return sampled_q_batch, sampled_y_batch

    # core

    def _median(self, net: torch.Tensor) -> torch.Tensor:
        return self._quantile(net, None)["y"]

    def _cdf(
        self,
        net: torch.Tensor,
        y_batch: torch.Tensor,
        need_optimize: bool,
        return_pdf: bool,
    ) -> tensor_dict_type:
        y_batch.requires_grad_(return_pdf)
        with mode_context(self, to_train=None, use_grad=return_pdf):
            y_batch_ = (y_batch - self.y_min) / (0.5 * self.y_diff) - 1.0
            results = self.core(net, y_batch=y_batch_)
            cdf = results["q"]
        if not return_pdf:
            pdf = None
        else:
            pdf = get_gradient(cdf, y_batch, need_optimize, need_optimize)
            assert isinstance(pdf, torch.Tensor)
            y_batch.requires_grad_(False)
        results["pdf"] = pdf
        return results

    def _quantile(
        self,
        net: torch.Tensor,
        q_batch: Optional[torch.Tensor],
    ) -> tensor_dict_type:
        if q_batch is None:
            results = self.core(net, median=True)
        else:
            q_batch = 2.0 * q_batch - 1.0
            results = self.core(net, q_batch=q_batch)
        results["y"] = (results["y"] + 1.0) * (0.5 * self.y_diff) + self.y_min
        return results

    def _core(self, net: torch.Tensor, **kwargs: Any) -> tensor_dict_type:
        # generate quantile / anchor batch
        with timing_context(self, "forward.generate_anchors"):
            if self.training:
                q_batch = np.random.random([len(net), 1]).astype(np.float32)
                y_batch = np.random.random([len(net), 1]).astype(np.float32)
            else:
                q_batch = y_batch = self.default_anchors
                n_repeat = int(len(net) / len(q_batch)) + 1
                q_batch = np.repeat(q_batch, n_repeat)[: len(net)]
                y_batch = np.repeat(y_batch, n_repeat)[: len(net)]
            y_batch = y_batch * self.y_diff + self.y_min
            q_batch = self._convert_np_anchors(q_batch)
            y_batch = self._convert_np_anchors(y_batch)
        # is synthetic
        is_synthetic = kwargs.get("synthetic", False)
        # build predictions
        with timing_context(self, "forward.median"):
            median = self._median(net)
        with timing_context(self, "forward.quantile"):
            quantile_results = self._quantile(net, q_batch)
            quantiles = quantile_results["y"]
            quantiles_full = quantile_results["y_full"]
            assert quantiles is not None and quantiles_full is not None
        with timing_context(self, "forward.cdf"):
            cdf_results = self._cdf(net, y_batch, True, True)
            cdf, pdf, cdf_full = map(cdf_results.get, ["q", "pdf", "q_full"])
            assert cdf is not None and pdf is not None and cdf_full is not None
        # build sampled predictions
        if not self.training or is_synthetic:
            sampled_q_batch = sampled_y_batch = None
            sampled_quantiles = sampled_cdf = sampled_pdf = None
            sampled_quantiles_full = sampled_cdf_full = None
        else:
            sampled_q_batch, sampled_y_batch = self._sample_anchors(len(net))
            with timing_context(self, "forward.sampled_quantile"):
                sampled_quantile_results = self._quantile(net, sampled_q_batch)
                sampled_quantiles = sampled_quantile_results["y"]
                sampled_quantiles_full = sampled_quantile_results["y_full"]
                assert sampled_quantiles is not None
                assert sampled_quantiles_full is not None
            with timing_context(self, "forward.sampled_pdf"):
                sampled_cdf_results = self._cdf(net, sampled_y_batch, True, True)
                sampled_cdf, sampled_pdf = map(sampled_cdf_results.get, ["q", "pdf"])
                sampled_cdf_full = sampled_cdf_results["q_full"]
                assert sampled_cdf is not None and sampled_pdf is not None
                assert sampled_cdf_full is not None
        # construct results
        return {
            "net": net,
            "predictions": median,
            "q_batch": q_batch,
            "y_batch": y_batch,
            "sampled_q_batch": sampled_q_batch,
            "sampled_y_batch": sampled_y_batch,
            "pdf": pdf,
            "cdf": cdf,
            "cdf_full": cdf_full,
            "quantiles": quantiles,
            "quantiles_full": quantiles_full,
            "sampled_pdf": sampled_pdf,
            "sampled_cdf": sampled_cdf,
            "sampled_cdf_full": sampled_cdf_full,
            "sampled_quantiles": sampled_quantiles,
            "sampled_quantiles_full": sampled_quantiles_full,
        }

    # API

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        # pre-processing
        x_batch = batch["x_batch"]
        net = self._split_features(x_batch, batch_indices, loader_name).merge()
        if self.tr_data.is_ts:
            net = net.view(net.shape[0], -1)
        # check inference
        forward_dict = {}
        predict_pdf = kwargs.get("predict_pdf", False)
        predict_cdf = kwargs.get("predict_cdf", False)
        if predict_pdf or predict_cdf:
            y = kwargs.get("y")
            if y is None:
                raise ValueError(f"pdf / cdf cannot be predicted without y")
            y_batch = self._expand(len(net), y, numpy=True)
            y_batch = self.tr_data.transform_labels(y_batch)
            y_batch = to_torch(y_batch).to(self.device)
            results = self._cdf(net, y_batch, False, predict_pdf)
            if predict_pdf:
                forward_dict["pdf"] = results["pdf"]
            if predict_cdf:
                forward_dict["cdf"] = results["q"]
        predict_quantile = kwargs.get("predict_quantiles")
        if predict_quantile:
            q = kwargs.get("q")
            if q is None:
                raise ValueError(f"quantile cannot be predicted without q")
            q_batch = self._expand(len(net), q)
            forward_dict["quantiles"] = self._quantile(net, q_batch)["y"]
        if not forward_dict:
            forward_dict = self._core(net, **shallow_copy_dict(kwargs))
        return forward_dict

    def loss_function(
        self,
        batch: tensor_dict_type,
        batch_indices: np.ndarray,
        forward_results: tensor_dict_type,
    ) -> tensor_dict_type:
        y_batch = batch["y_batch"]
        losses, losses_dict = self.loss(forward_results, y_batch)
        net = forward_results["net"]
        if (
            self.training
            and self._synthetic_step > 0
            and self._step_count % self._synthetic_step == 0
        ):
            with timing_context(self, "synthetic.get_batch"):
                net_min = torch.min(net, dim=0)[0].view(*net.shape[1:])
                net_max = torch.max(net, dim=0)[0].view(*net.shape[1:])
                net_diff = net_max - net_min
                lower_bound = 0.5 * (self._synthetic_range - 1) * net_diff
                synthetic_net = net.new_empty(net.shape)
                synthetic_net.uniform_(0, 1)
                synthetic_net = synthetic_net * net_diff * self._synthetic_range
                synthetic_net = synthetic_net - (lower_bound - net_min)
            with timing_context(self, "synthetic.forward"):
                synthetic_outputs = self._core(synthetic_net, synthetic=True)
            with timing_context(self, "synthetic.loss"):
                synthetic_losses, _ = self.loss._core(  # type: ignore
                    synthetic_outputs,
                    y_batch,
                    check_monotonous_only=True,
                )
            losses_dict["synthetic"] = synthetic_losses
            losses = losses + synthetic_losses
        losses_dict["loss"] = losses
        losses_dict = {k: v.mean() for k, v in losses_dict.items()}
        if self.training:
            self._step_count += 1
        else:
            assert isinstance(y_batch, torch.Tensor)
            q_losses = []
            y_batch = to_numpy(y_batch)
            for q in self._quantile_anchors:
                self.q_metric.config["q"] = q
                yq = self._quantile(net, self._expand(len(net), q))["y"]
                q_losses.append(self.q_metric.metric(y_batch, to_numpy(yq)))
            quantile_metric = -sum(q_losses) / len(q_losses) * self.q_metric.sign
            ddr_loss = torch.tensor([quantile_metric], dtype=torch.float32)
            losses_dict["ddr"] = ddr_loss
        return losses_dict


__all__ = ["DDR"]
