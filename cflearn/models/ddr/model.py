import torch

import numpy as np
import torch.nn as nn

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
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
        to_latent = instance.config.setdefault("to_latent", True)
        latent_dim = instance.config.setdefault("latent_dim", None)
        num_blocks = instance.config.setdefault("num_blocks", None)

        def builder(latent_dim_: int) -> nn.Module:
            h_dim = latent_dim_ // 2
            return MLP.simple(h_dim, None, [h_dim], activation="Tanh")

        transition_builder = instance.config.setdefault("transition_builder", builder)
        cfg.update(
            {
                "num_blocks": num_blocks,
                "to_latent": to_latent,
                "latent_dim": latent_dim,
                "transition_builder": transition_builder,
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
        anchors = np.asarray(y_anchors, np.float32)
        self._anchor_choice_array = y_min + self.y_diff * anchors
        self._quantile_anchors = np.asarray(q_anchors, np.float32)
        self._synthetic_range = self.config.setdefault("synthetic_range", 5)
        # loss config
        self._loss_config = self.config.setdefault("loss_config", {})
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

    @property
    def q_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda q: 2.0 * q - 1.0

    @property
    def q_inv_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda q: 0.5 * (q + 1.0)

    @property
    def y_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda y: (y - self.y_min) / (0.5 * self.y_diff) - 1.0

    @property
    def y_inv_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda y: (y + 1.0) * (0.5 * self.y_diff) + self.y_min

    def _quantile(
        self,
        latent: torch.Tensor,
        q_batch: Optional[torch.Tensor],
        do_inverse: bool,
    ) -> tensor_dict_type:
        if q_batch is None:
            results = self.core(
                latent,
                median=True,
                do_inverse=do_inverse,
            )
        else:
            q_batch = self.q_fn(q_batch)
            results = self.core(
                latent,
                q_batch=q_batch,
                do_inverse=do_inverse,
            )
        results["y"] = self.y_inv_fn(results["y"])
        if do_inverse:
            results["q_inverse"] = self.q_inv_fn(results["q_inverse"])
        return results

    def _median(
        self,
        latent: torch.Tensor,
        do_inverse: bool,
    ) -> tensor_dict_type:
        return self._quantile(latent, None, do_inverse)

    def _cdf(
        self,
        latent: torch.Tensor,
        y_batch: torch.Tensor,
        need_optimize: bool,
        return_pdf: bool,
        do_inverse: bool,
    ) -> tensor_dict_type:
        y_batch.requires_grad_(return_pdf)
        with mode_context(self, to_train=None, use_grad=return_pdf):
            y_batch_ = self.y_fn(y_batch)
            results = self.core(
                latent,
                y_batch=y_batch_,
                do_inverse=do_inverse,
            )
            cdf = results["q"] = self.q_inv_fn(results["q"])
            if do_inverse:
                results["y_inverse"] = self.y_inv_fn(results["y_inverse"])
        if not return_pdf:
            pdf = None
        else:
            pdf = get_gradient(cdf, y_batch, need_optimize, need_optimize)
            assert isinstance(pdf, torch.Tensor)
            y_batch.requires_grad_(False)
        results["pdf"] = pdf
        return results

    def _core(self, latent: torch.Tensor, synthetic: bool) -> tensor_dict_type:
        batch_size = len(latent)
        # generate quantile / anchor batch
        with timing_context(self, "forward.generate_anchors"):
            if self.training:
                q_batch = np.random.random([batch_size, 1]).astype(np.float32)
                y_batch = np.random.random([batch_size, 1]).astype(np.float32)
            else:
                q_batch = y_batch = self.default_anchors
                n_repeat = int(batch_size / len(q_batch)) + 1
                q_batch = np.repeat(q_batch, n_repeat)[:batch_size]
                y_batch = np.repeat(y_batch, n_repeat)[:batch_size]
            y_batch = self.y_min + self.y_diff * y_batch
            q_batch = self._convert_np_anchors(q_batch)
            y_batch = self._convert_np_anchors(y_batch)
        # build predictions
        with timing_context(self, "forward.median"):
            if synthetic:
                median = median_inverse = None
            else:
                results = self._median(latent, True)
                median = results["y"]
                median_inverse = results["q_inverse"]
        with timing_context(self, "forward.quantile"):
            quantile_results = self._quantile(latent, q_batch, True)
            y = quantile_results["y"]
            q_inverse = quantile_results["q_inverse"]
            assert y is not None and q_inverse is not None
        with timing_context(self, "forward.cdf"):
            cdf_results = self._cdf(latent, y_batch, True, True, True)
            cdf, pdf, y_inverse = map(cdf_results.get, ["q", "pdf", "y_inverse"])
            assert cdf is not None and pdf is not None and y_inverse is not None
        # build sampled predictions
        if not self.training:
            sampled_q_batch = sampled_y_batch = None
            sampled_y = sampled_cdf = sampled_pdf = None
            sampled_q_inverse = sampled_y_inverse = None
        else:
            sampled_q_batch, sampled_y_batch = self._sample_anchors(batch_size)
            with timing_context(self, "forward.sampled_quantile"):
                sq_results = self._quantile(latent, sampled_q_batch, True)
                sampled_q_inverse = sq_results["q_inverse"]
                assert sampled_q_inverse is not None
                if synthetic:
                    sampled_y = None
                else:
                    sampled_y = sq_results["y"]
                    assert sampled_y is not None
            with timing_context(self, "forward.sampled_cdf"):
                sy_results = self._cdf(latent, sampled_y_batch, True, True, True)
                sampled_pdf = sy_results["pdf"]
                sampled_y_inverse = sy_results["y_inverse"]
                assert sampled_pdf is not None
                assert sampled_y_inverse is not None
                if synthetic:
                    sampled_cdf = None
                else:
                    sampled_cdf, sampled_pdf = map(sy_results.get, ["q", "pdf"])
                    assert sampled_cdf is not None
        # construct results
        return {
            "latent": latent,
            "predictions": median,
            "median_inverse": median_inverse,
            "q_batch": q_batch,
            "y_batch": y_batch,
            "sampled_q_batch": sampled_q_batch,
            "sampled_y_batch": sampled_y_batch,
            "y": y,
            "q_inverse": q_inverse,
            "pdf": pdf,
            "cdf": cdf,
            "y_inverse": y_inverse,
            "sampled_y": sampled_y,
            "sampled_q_inverse": sampled_q_inverse,
            "sampled_cdf": sampled_cdf,
            "sampled_pdf": sampled_pdf,
            "sampled_y_inverse": sampled_y_inverse,
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
        latent = self.core.get_latent(net)
        # check inference
        forward_dict = {}
        predict_pdf = kwargs.get("predict_pdf", False)
        predict_cdf = kwargs.get("predict_cdf", False)
        predict_quantile = kwargs.get("predict_quantiles", False)
        if predict_pdf or predict_cdf:
            y = kwargs.get("y")
            if y is None:
                raise ValueError(f"pdf / cdf cannot be predicted without y")
            y_batch = self._expand(len(latent), y, numpy=True)
            y_batch = self.tr_data.transform_labels(y_batch)
            y_batch = to_torch(y_batch).to(self.device)
            results = self._cdf(latent, y_batch, False, predict_pdf, False)
            if predict_pdf:
                forward_dict["pdf"] = results["pdf"]
            if predict_cdf:
                forward_dict["cdf"] = results["q"]
        if predict_quantile:
            q = kwargs.get("q")
            if q is None:
                raise ValueError(f"quantile cannot be predicted without q")
            q_batch = self._expand(len(latent), q)
            forward_dict["quantiles"] = self._quantile(latent, q_batch, False)["y"]
        if not forward_dict:
            forward_dict = self._core(latent, False)
        forward_dict["net"] = net
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
            with timing_context(self, "synthetic.forward"):
                synthetic_net = net.detach() * self._synthetic_range
                synthetic_latent = self.core.get_latent(synthetic_net)
                synthetic_outputs = self._core(synthetic_latent, True)
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
        if self.training:
            self._step_count += 1
        else:
            assert isinstance(y_batch, torch.Tensor)
            q_losses = []
            y_batch = to_numpy(y_batch)
            latent = forward_results["latent"]
            for q in self._quantile_anchors:
                self.q_metric.config["q"] = q
                yq = self._quantile(latent, self._expand(len(latent), q), False)["y"]
                q_losses.append(self.q_metric.metric(y_batch, to_numpy(yq)))
            quantile_metric = -sum(q_losses) / len(q_losses) * self.q_metric.sign
            ddr_loss = torch.tensor([quantile_metric], dtype=torch.float32)
            losses_dict["ddr"] = ddr_loss
        return losses_dict


__all__ = ["DDR"]
