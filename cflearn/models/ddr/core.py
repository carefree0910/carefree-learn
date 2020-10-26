import torch

import numpy as np
import torch.nn as nn

from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from typing import NamedTuple
from torch.optim import Optimizer
from cftool.ml import Metrics
from cftool.misc import is_numeric
from cftool.misc import update_dict
from cftool.misc import timing_context
from cftool.misc import shallow_copy_dict
from cfdata.tabular import DataLoader
from cfdata.tabular import TabularData
from ...types import tensor_dict_type

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None

from .loss import DDRLoss
from ..fcnn.core import FCNN
from ...misc.toolkit import *
from ...modules.blocks import *

proj_type = Optional[Union[Linear, MLP]]
tensors_type = Union[
    torch.Tensor,
    List[torch.Tensor],
    Dict[str, Optional[Union[torch.Tensor, tensor_dict_type]]],
]


class BasicInfo(NamedTuple):
    median: torch.Tensor
    median_detach: torch.Tensor
    feature_layers: List[torch.Tensor]
    anchor_batch: Optional[torch.Tensor]
    quantile_batch: Optional[torch.Tensor]


@FCNN.register("ddr")
class DDR(FCNN):
    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        tr_loader: DataLoader,
        cv_data: TabularData,
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
            cv_data,
            tr_weights,
            cv_weights,
            device,
            use_tqdm=use_tqdm,
        )
        self.__feature_params: List[nn.Parameter] = []
        self.__reg_params: List[nn.Parameter] = []
        self._inject_median_params()
        self._init_cdf_layers()
        self._init_quantile_layers()
        self._init_parameters()
        self.q_metric = Metrics("quantile")
        self._step_count = 0

    def _init_config(self) -> None:
        self.config.setdefault("ema_decay", 0.999)
        # common mapping configs
        self._common_configs = self.config.setdefault("common_configs", {})
        self._common_configs.setdefault("pruner_config", None)
        self._common_configs.setdefault("batch_norm", False)
        self._common_configs.setdefault("dropout", 0.0)
        # feature mappings
        self.feature_units = self.config.setdefault("feature_units", [512, 512])
        mapping_configs = self.config.setdefault("mapping_configs", {})
        if isinstance(mapping_configs, dict):
            mapping_configs.setdefault("activation", "mish")
            mapping_configs.setdefault("init_method", "xavier_uniform")
            mapping_configs.setdefault("batch_norm", False)
            mapping_configs.setdefault("dropout", 0.0)
            mapping_configs.setdefault("bias", False)
            mapping_configs = [mapping_configs] * self.num_feature_layers
        self.median_units = self.config.setdefault("median_units", [512])
        self._q_reg_activation = self.config.setdefault(
            "quantile_reg_activation", "ReLU"
        )
        self._reg_init = self.config.setdefault(
            "regression_initialization", "xavier_uniform"
        )
        median_mapping_configs = self.config.setdefault(
            "median_mapping_configs", shallow_copy_dict(self._common_configs)
        )
        if isinstance(median_mapping_configs, dict):
            median_mapping_configs.setdefault("activation", self._q_reg_activation)
            median_mapping_configs.setdefault("init_method", self._reg_init)
            median_mapping_configs = [median_mapping_configs] * len(self.median_units)
        self.config["hidden_units"] = self.feature_units + self.median_units
        self.config["mapping_configs"] = mapping_configs + median_mapping_configs
        # inherit
        super()._init_config()
        # common
        self._joint_training = self.config.setdefault("joint_training", True)
        self._use_gradient_loss = self.config.setdefault("use_gradient_loss", True)
        fetches = self.config.setdefault("fetches", "all")
        if fetches == "all":
            self._fetches = {"cdf", "quantile"}
        elif isinstance(fetches, str):
            self._fetches = {fetches}
        else:
            self._fetches = set(fetches)

        cdf_ratio_anchors = quantile_anchors = self.default_anchors

        if not self.fetch_cdf:
            cdf_ratio_anchors = None
        if not self.fetch_quantile:
            quantile_anchors = None
        cdf_ratio_anchors = self.config.setdefault(
            "cdf_ratio_anchors", cdf_ratio_anchors
        )
        quantile_anchors = self.config.setdefault("quantile_anchors", quantile_anchors)
        labels = self.tr_data.processed.y

        y_min, y_max = labels.min(), labels.max()
        self.y_min, self.y_max = y_min, y_max
        self.y_diff = y_max - y_min
        if cdf_ratio_anchors is None:
            self._anchor_choice_array = None
        else:
            anchors = np.asarray(cdf_ratio_anchors, np.float32)
            self._anchor_choice_array = y_min + (y_max - y_min) * anchors
        if quantile_anchors is None:
            self._quantile_anchors = None
        else:
            self._quantile_anchors = np.asarray(quantile_anchors, np.float32)
        # use negative value to disable dual inference
        # use 0 (sample) for stable performance & 1 (gradient descent) for faster (and maybe better) performance
        self._dual_inference_version = self.config.setdefault(
            "dual_inference_version", -1
        )
        self._synthetic_range = self.config.setdefault("synthetic_range", 3)
        # loss config
        self._loss_config = self.config.setdefault("loss_config", {})
        self._loss_config.setdefault("use_dynamic_weights", False)
        self._loss_config.setdefault("use_anneal", self.num_feature_layers > 0)
        self._loss_config.setdefault("mtl_method", None)
        self._loss_config["joint_training"] = self._joint_training
        # trainer config
        default_metric_types = (
            ["ddr", "loss"] if self.fetch_quantile else ["mae", "loss"]
        )
        trainer_config = self._pipeline_config.setdefault("trainer_config", {})
        trainer_config = update_dict(
            trainer_config,
            {
                "clip_norm": 1.0,
                "num_epoch": 40,
                "max_epoch": 1000,
                "batch_size": 128,
                "metric_config": {"types": default_metric_types, "decay": 0.5},
            },
        )
        num_train_samples = self.tr_data.processed.x.shape[0]
        num_epoch = trainer_config["num_epoch"]
        batch_size = trainer_config["batch_size"]
        anneal_step = self._loss_config.setdefault(
            "anneal_step", (num_train_samples * num_epoch) // (batch_size * 2)
        )
        self._loss_config.setdefault("anneal_step", anneal_step)
        self._pipeline_config["trainer_config"] = trainer_config
        self._trainer_config = trainer_config
        # optimize schema
        self._reg_step = int(self.config.setdefault("reg_step", 10))
        self._feature_step = int(self.config.setdefault("feature_step", 5))
        self._synthetic_step = int(self.config.setdefault("synthetic_step", 5))

    def _init_loss(self) -> None:
        loss_config = shallow_copy_dict(self._loss_config)
        loss_config["device"] = self.device
        self.loss = DDRLoss(loss_config, "none")

    def _init_mlp_config(
        self,
        prefix: str,
        default_bias: bool,
        default_activation: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        activation = self.config.setdefault(f"{prefix}_activation", default_activation)
        mapping_configs_key = f"{prefix}_mapping_configs"
        mapping_configs = self.config.setdefault(
            mapping_configs_key, shallow_copy_dict(self._common_configs)
        )
        if isinstance(mapping_configs, dict):
            mapping_configs.setdefault("bias", default_bias)
            mapping_configs.setdefault("activation", activation)
        final_mapping_config = self.config.setdefault(
            f"{prefix}_final_mapping_config", {}
        )
        return mapping_configs, final_mapping_config

    def _inject_median_params(self) -> None:
        for mapping in self.mlp.mappings[: self.num_feature_layers]:
            self.__feature_params.extend(mapping.parameters())
        for mapping in self.mlp.mappings[self.num_feature_layers :]:
            self.__reg_params.extend(mapping.parameters())

    def _init_cdf_layers(self) -> None:
        self.cdf_reg: proj_type
        self.cdf_feature_projection: proj_type
        if not self.fetch_cdf:
            self.cdf_reg = self.cdf_feature_projection = None
            return None
        cdf_input_dim = self.merged_dim + 1
        if not self.no_joint_features:
            cdf_input_dim += self.feature_dim
        # regression part
        cdf_reg_units = self.config.setdefault("cdf_reg_units", [512])
        if not cdf_reg_units:
            self.cdf_reg = Linear(cdf_input_dim, 1, **self._common_configs)
        else:
            bundle = self._init_mlp_config("cdf_reg", True, "ReLU")
            mapping_configs, final_mapping_config = bundle
            self.cdf_reg = self._make_projection(
                cdf_input_dim,
                1,
                cdf_reg_units,
                mapping_configs,
                final_mapping_config,
            )
        self.__reg_params.extend(self.cdf_reg.parameters())
        # feature part
        if self.no_joint_features:
            self.cdf_feature_projection = None
        else:
            cdf_feature_units = self.feature_units.copy()
            bundle = self._init_mlp_config("cdf_feature", False, "ReLU")
            mapping_configs, final_mapping_config = bundle
            self.cdf_feature_projection = self._make_projection(
                1,
                None,
                cdf_feature_units,
                mapping_configs,
                final_mapping_config,
            )
            self.__feature_params.extend(self.cdf_feature_projection.parameters())

    def _init_quantile_layers(self) -> None:
        if not self.fetch_quantile:
            self.median_residual_reg = None
            self.aq_res_reg = self.mq_res_reg = None
            return
        quantile_input_dim = self.merged_dim
        if self.num_feature_layers > 0:
            quantile_input_dim += self.feature_dim
        activations = Activations()
        self.mish = activations.mish
        self.relu = activations.ReLU
        # median residual part
        median_residual_units = self.config.setdefault("median_residual_units", [512])
        bundle = self._init_mlp_config("median_residual", True, self._q_reg_activation)
        mapping_configs, final_mapping_config = bundle
        self.median_residual_reg = self._make_projection(
            quantile_input_dim,
            2,
            median_residual_units,
            mapping_configs,
            final_mapping_config,
        )
        self.__reg_params.extend(self.median_residual_reg.parameters())
        # quantile residual regression part
        qr_input_dim = quantile_input_dim + 1
        qr_reg_units = self.config.setdefault("quantile_res_reg_units", [512])
        mapping_configs, final_mapping_config = self._init_mlp_config(
            "quantile_res_reg", True, self._q_reg_activation
        )
        q_res_reg_args = (
            qr_input_dim,
            2,
            qr_reg_units,
            mapping_configs,
            final_mapping_config,
        )
        self.aq_res_reg = self._make_projection(*q_res_reg_args)
        self.mq_res_reg = self._make_projection(*q_res_reg_args)
        self.__reg_params.extend(self.aq_res_reg.parameters())
        self.__reg_params.extend(self.mq_res_reg.parameters())
        # feature part
        if self.no_joint_features:
            self.aq_proj = self.mq_proj = None
        else:
            quantile_feature_units = self.feature_units.copy()
            mapping_configs, final_mapping_config = self._init_mlp_config(
                "quantile_feature", False, "mish"
            )
            q_feature_proj_args = (
                1,
                None,
                quantile_feature_units,
                mapping_configs,
                final_mapping_config,
            )
            self.aq_proj = self._make_projection(*q_feature_proj_args)  # type: ignore
            self.mq_proj = self._make_projection(*q_feature_proj_args)  # type: ignore
            self.__feature_params.extend(self.aq_proj.parameters())
            self.__feature_params.extend(self.mq_proj.parameters())

    def _init_parameters(self) -> None:
        all_parameters = set(self.parameters())
        feature_params = set(self.__feature_params)
        reg_params = set(self.__feature_params)
        self.__base_params = list(all_parameters - feature_params - reg_params)
        base_config = {"optimizer": "adam", "optimizer_config": {"lr": 3.5e-4}}
        optimizers = {"all": base_config}
        if self._feature_step > 0 and feature_params:
            optimizers["feature_parameters"] = shallow_copy_dict(base_config)
        if self._reg_step > 0:
            optimizers["reg_parameters"] = shallow_copy_dict(base_config)
        if self.__base_params:
            optimizers["base_parameters"] = shallow_copy_dict(base_config)
        trainer_optimizers = self._trainer_config.setdefault("optimizers", {})
        trainer_optimizers = update_dict(trainer_optimizers, optimizers)
        self._trainer_config["optimizers"] = trainer_optimizers

    def _optimizer_step(
        self,
        optimizers: Dict[str, Optimizer],
        grad_scalar: Optional["amp.GradScaler"],  # type: ignore
    ) -> None:
        for key in self.target_parameters:
            opt = optimizers[key]
            if grad_scalar is None:
                opt.step()
            else:
                grad_scalar.step(opt)
                grad_scalar.update()
            opt.zero_grad()

    @staticmethod
    def _make_projection(
        in_dim: int,
        out_dim: Optional[int],
        units: List[int],
        mapping_configs: Union[Dict[str, Any], List[Dict[str, Any]]],
        final_mapping_config: Dict[str, Any],
    ) -> Union[Linear, MLP]:
        if not units:
            if out_dim is None:
                raise ValueError("either `out_dim` or `units` should be provided")
            return Linear(in_dim, out_dim, **final_mapping_config)
        return MLP(
            in_dim,
            out_dim,
            units,
            mapping_configs,
            final_mapping_config=final_mapping_config,
        )

    @property
    def feature_dim(self) -> int:
        return self.feature_units[-1]

    @property
    def num_feature_layers(self) -> int:
        return len(self.feature_units)

    @property
    def no_joint_features(self) -> bool:
        return self.num_feature_layers == 0

    @property
    def quantile_residual_core_keys(self) -> List[str]:
        return ["median_residual", "quantile_residual", "quantile_sign"]

    @property
    def default_anchors(self) -> np.ndarray:
        return np.linspace(0.05, 0.95, 10).astype(np.float32)

    @property
    def fetch_cdf(self) -> bool:
        return "cdf" in self._fetches

    @property
    def fetch_quantile(self) -> bool:
        return "quantile" in self._fetches

    @property
    def trigger_feature_update(self) -> bool:
        step = self._feature_step > 0 and self._step_count % self._feature_step == 0
        has_features = len(self.__feature_params) > 0
        return step and has_features

    @property
    def trigger_reg_update(self) -> bool:
        return self._reg_step > 0 and self._step_count % self._reg_step == 0

    @property
    def base_parameters(self) -> List[nn.Parameter]:
        return self.__base_params

    @property
    def feature_parameters(self) -> List[nn.Parameter]:
        return self.__feature_params

    @property
    def reg_parameters(self) -> List[nn.Parameter]:
        return self.__reg_params

    @property
    def target_parameters(self) -> Set[str]:
        if self.trigger_feature_update or self.trigger_reg_update:
            if not self.trigger_feature_update:
                return {"reg_parameters"}
            if not self.trigger_reg_update:
                return {"feature_parameters"}
            return {"feature_parameters", "reg_parameters"}
        return {"all"}

    # build outputs

    @staticmethod
    def _merge_responses(
        anchors: torch.Tensor,
        projection: proj_type,
        feature_layers: List[torch.Tensor],
    ) -> torch.Tensor:
        if projection is None:
            return feature_layers[-1]
        net = anchors
        if isinstance(projection, Linear):
            net = projection(net)
        else:
            for i, mapping in enumerate(projection.mappings):
                if i > 0:
                    net = net + feature_layers[i]
                net = mapping(net)
        return net

    def _build_cdf(
        self,
        feature_layers: List[torch.Tensor],
        anchor_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.cdf_reg is not None
        anchor_batch_ratio = self._get_anchor_batch_ratio(anchor_batch)
        net = self._merge_responses(
            anchor_batch_ratio,
            self.cdf_feature_projection,
            feature_layers,
        )
        concat_list = [net, anchor_batch_ratio]
        if not self.no_joint_features:
            concat_list.append(feature_layers[0])
        features = torch.cat(concat_list, dim=1)
        cdf_raw = self.cdf_reg(features)
        cdf = torch.sigmoid(cdf_raw)
        return cdf, cdf_raw

    def _build_pdf(
        self,
        feature_layers: List[torch.Tensor],
        anchor_batch: torch.Tensor,
        need_optimization: bool,
    ) -> Tuple[torch.Tensor, ...]:
        anchor_batch.requires_grad_(True)
        with mode_context(self, to_train=None, use_grad=True):
            cdf, cdf_raw = self._build_cdf(feature_layers, anchor_batch)
        pdf = get_gradient(cdf, anchor_batch, need_optimization, need_optimization)
        assert isinstance(pdf, torch.Tensor)
        anchor_batch.requires_grad_(False)
        return cdf, cdf_raw, pdf

    def _build_median_residual(
        self,
        feature_layers: List[torch.Tensor],
        quantile_batch: Optional[torch.Tensor],
        sign: Optional[torch.Tensor] = None,
    ) -> tensor_dict_type:
        assert self.median_residual_reg is not None
        concat_list = [feature_layers[-1]]
        if self.num_feature_layers > 0:
            concat_list.append(feature_layers[0])
        mr_features = torch.cat(concat_list, dim=1)
        mr = self.median_residual_reg(mr_features)
        mr = self.mish(mr)
        mr_pos, mr_neg = torch.chunk(mr, 2, dim=1)
        mr_neg = -mr_neg
        if sign is None:
            assert quantile_batch is not None
            sign = torch.sign(quantile_batch - 0.5)
        pos_quantile_mask = sign == 1.0
        median_residual = torch.where(pos_quantile_mask, mr_pos, mr_neg)
        return {
            "quantile_sign": sign,
            "median_residual": median_residual,
            "pos_quantile_mask": pos_quantile_mask,
        }

    def _build_quantile_residual(
        self,
        feature_layers: List[torch.Tensor],
        quantile_batch: torch.Tensor,
        pressure_batch: bool = False,
        *,
        keys: Optional[Union[str, List[str]]] = None,
    ) -> tensors_type:
        add_ratio, mul_ratio = self._get_quantile_batch_ratio(quantile_batch)
        add_net = self._merge_responses(add_ratio, self.aq_proj, feature_layers)
        mul_net = self._merge_responses(mul_ratio, self.mq_proj, feature_layers)
        concat_list = []
        if self.num_feature_layers > 0:
            concat_list.append(feature_layers[0])
        sub_quantile_dict, sub_quantile_pos_dict, sub_quantile_neg_dict = {}, {}, {}
        mr_results = self._build_median_residual(feature_layers, quantile_batch)
        median_residual = mr_results["median_residual"]
        q_sign, q_pos_mask = map(mr_results.get, ["quantile_sign", "pos_quantile_mask"])
        for res_type, q_features, q_ratio, q_res_reg, activation in zip(
            ["add", "mul"],
            [add_net, mul_net],
            [add_ratio, mul_ratio],
            [self.aq_res_reg, self.mq_res_reg],
            [self.mish, self.relu],
        ):
            assert q_res_reg is not None
            assert q_pos_mask is not None
            local_concat_list = concat_list + [q_features, q_ratio]
            local_quantile_features = torch.cat(local_concat_list, dim=1)
            sq = q_res_reg(local_quantile_features)
            sq = activation(sq)
            sub_quantile_pos, sub_quantile_neg = torch.chunk(sq, 2, dim=1)
            if res_type == "add":
                sub_quantile_neg = -sub_quantile_neg
            if pressure_batch:
                sub_quantile_pos_dict[res_type] = sub_quantile_pos
                sub_quantile_neg_dict[res_type] = sub_quantile_neg
                continue
            sub_quantile_dict[res_type] = torch.where(
                q_pos_mask,
                sub_quantile_pos,
                sub_quantile_neg,
            )
        if pressure_batch:
            return {
                "pp_dict": sub_quantile_pos_dict,
                "pn_dict": sub_quantile_neg_dict,
            }
        quantile_residual = (
            median_residual.detach() * sub_quantile_dict["mul"]
            + sub_quantile_dict["add"]
        )
        rs = {
            "quantile_sign": q_sign,
            "median_residual": median_residual,
            "quantile_residual": quantile_residual,
            "additive_quantile_features": add_net,
            "multiply_quantile_features": mul_net,
        }
        if keys is None:
            return rs
        if isinstance(keys, str):
            value = rs[keys]
            assert isinstance(value, torch.Tensor)
            return value
        values = [rs[key] for key in keys]
        assert all(map(lambda v: v is not None, values))
        return values  # type: ignore

    # fetch & combine outputs

    def _get_anchor_batch_ratio(self, anchor_batch: torch.Tensor) -> torch.Tensor:
        return (anchor_batch - self.y_min) / self.y_diff

    @staticmethod
    def _get_quantile_batch_ratio(
        quantile_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        add_ratio = 4 * torch.where(
            quantile_batch > 0.5, quantile_batch - 0.75, 0.25 - quantile_batch
        )
        mul_ratio = 2 * torch.abs(quantile_batch - 0.5)
        return add_ratio, mul_ratio

    def _convert_np_anchors(
        self,
        np_anchors: Optional[np.ndarray],
    ) -> Optional[torch.Tensor]:
        if np_anchors is None:
            return None
        tensor = to_torch(np_anchors.reshape([-1, 1]))
        return tensor.to(self.device).requires_grad_(True)

    def _get_feature_layers(self, net: torch.Tensor) -> List[torch.Tensor]:
        feature_layers = [net]
        for mapping in self.mlp.mappings[: self.num_feature_layers]:
            net = mapping(net)
            feature_layers.append(net)
        return feature_layers

    def _get_median(self, feature_layers: List[torch.Tensor]) -> torch.Tensor:
        net = feature_layers[-1]
        for mapping in self.mlp.mappings[self.num_feature_layers :]:
            net = mapping(net)
        return net

    def _get_distribution(
        self,
        basic_info: BasicInfo,
        fetch_anchors: bool,
    ) -> tensor_dict_type:
        if not self.fetch_cdf:
            raise ValueError("distribution could not be fetched")
        anchor_batch = basic_info.anchor_batch
        feature_layers = basic_info.feature_layers
        assert anchor_batch is not None
        cdf, cdf_raw, pdf = self._build_pdf(feature_layers, anchor_batch, True)
        # get specific sampled distribution
        if not fetch_anchors or self._anchor_choice_array is None:
            sampled_anchors = sampled_cdf = sampled_cdf_raw = sampled_pdf = None
        else:
            num_choices = len(self._anchor_choice_array)
            choice_indices = np.random.randint(0, num_choices, len(cdf))
            chosen = self._anchor_choice_array[choice_indices]
            sampled_anchors = self._convert_np_anchors(chosen)
            assert sampled_anchors is not None
            sampled_cdf, sampled_cdf_raw, sampled_pdf = self._build_pdf(
                feature_layers, sampled_anchors, True
            )
        return {
            "cdf": cdf,
            "pdf": pdf,
            "cdf_raw": cdf_raw,
            "sampled_anchors": sampled_anchors,
            "sampled_cdf": sampled_cdf,
            "sampled_pdf": sampled_pdf,
            "sampled_cdf_raw": sampled_cdf_raw,
        }

    def _get_quantile_residual(
        self,
        basic_info: BasicInfo,
        fetch_anchors: bool,
    ) -> tensor_dict_type:
        if not self.fetch_quantile:
            raise ValueError("quantile could not be fetched")
        quantile_batch = basic_info.quantile_batch
        feature_layers = basic_info.feature_layers
        assert quantile_batch is not None
        quantile_dict = self._build_quantile_residual(feature_layers, quantile_batch)
        assert isinstance(quantile_dict, dict)
        # get median pressure
        with torch.no_grad():
            median_pressure_batch = quantile_batch.new_empty(
                quantile_batch.shape
            ).fill_(0.5)
        median_pressure_batch.requires_grad_(True)
        pressure_dict = self._build_quantile_residual(
            feature_layers,
            median_pressure_batch,
            pressure_batch=True,
        )
        assert isinstance(pressure_dict, dict)
        quantile_dict.update(pressure_dict)
        # get specific sampled quantile
        if not fetch_anchors or self._quantile_anchors is None or not self.training:
            sampled_quantiles = smr = sqr = sqs = None
        else:
            choice_indices = np.random.randint(
                0, len(self._quantile_anchors), len(quantile_batch)
            )
            chosen = self._quantile_anchors[choice_indices]
            sampled_quantiles = self._convert_np_anchors(chosen)
            assert sampled_quantiles is not None
            smr, sqr, sqs = self._build_quantile_residual(
                feature_layers,
                sampled_quantiles,
                keys=self.quantile_residual_core_keys,
            )
        quantile_dict.update(
            {
                "sampled_quantiles": sampled_quantiles,
                "sampled_median_residual": smr,
                "sampled_quantile_residual": sqr,
                "sampled_quantile_sign": sqs,
            }
        )
        return quantile_dict

    def _get_dual_cdf(
        self,
        basic_info: BasicInfo,
        quantile_residual: torch.Tensor,
    ) -> tensor_dict_type:
        median_detach = basic_info.median_detach
        feature_layers = basic_info.feature_layers
        dual_cdf, dual_cdf_raw = self._build_cdf(
            feature_layers, quantile_residual + median_detach
        )
        cmr, cdf_quantile_residual, cqs = self._build_quantile_residual(
            feature_layers,
            dual_cdf,
            keys=self.quantile_residual_core_keys,
        )
        return {
            "dual_cdf": dual_cdf,
            "dual_cdf_raw": dual_cdf_raw,
            "cdf_median_residual": cmr,
            "cdf_quantile_residual": cdf_quantile_residual,
            "cdf_quantile_sign": cqs,
        }

    def _get_dual_quantile(
        self,
        basic_info: BasicInfo,
        cdf: torch.Tensor,
    ) -> tensor_dict_type:
        median_detach = basic_info.median_detach
        feature_layers = basic_info.feature_layers
        dmr, dual_quantile_residual, dqs = self._build_quantile_residual(
            feature_layers,
            cdf,
            keys=self.quantile_residual_core_keys,
        )
        dual_quantile = median_detach + dual_quantile_residual
        quantile_cdf, quantile_cdf_raw = self._build_cdf(feature_layers, dual_quantile)
        return {
            "dual_median_residual": dmr,
            "dual_quantile": dual_quantile,
            "dual_quantile_residual": dqs,
            "quantile_cdf": quantile_cdf,
            "quantile_cdf_raw": quantile_cdf_raw,
        }

    # predict methods

    def _predict_cdf(self, init: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        feature_layers = self._get_feature_layers(init)
        cdf = self._build_cdf(feature_layers, y_batch)[0]
        return cdf

    def _predict_pdf(self, init: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        feature_layers = self._get_feature_layers(init)
        return self._build_pdf(feature_layers, y_batch, False)[-1]

    def _predict_quantile(
        self,
        init: torch.Tensor,
        q_batch: torch.Tensor,
    ) -> torch.Tensor:
        feature_layers = self._get_feature_layers(init)
        median = self._get_median(feature_layers)
        quantile = median + self._build_quantile_residual(
            feature_layers,
            q_batch,
            keys="quantile_residual",
        )
        return quantile

    def _predict_median_residual(
        self,
        init: torch.Tensor,
        sign_batch: torch.Tensor,
    ) -> torch.Tensor:
        feature_layers = self._get_feature_layers(init)
        median = self._get_median(feature_layers)
        predictions = self._build_median_residual(feature_layers, None, sign_batch)
        mr = predictions["median_residual"]
        residual = median + mr
        return residual

    # Core

    def _switch_training_status(self) -> None:
        target = self.target_parameters
        if "all" in target:
            self._switch_requires_grad(self.__base_params, True)
            self._switch_requires_grad(self.__feature_params, True)
            self._switch_requires_grad(self.__reg_params, True)
        else:
            self._switch_requires_grad(self.__base_params, False)
            self._switch_requires_grad(
                self.__feature_params, "feature_parameters" in target
            )
            self._switch_requires_grad(self.__reg_params, "reg_parameters" in target)

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

    def _core(self, init: torch.Tensor, **kwargs: Any) -> tensor_dict_type:
        # median only
        if kwargs.get("median_only", False):
            with timing_context(self, "forward.no_loss"):
                with eval_context(self):
                    feature_layers = self._get_feature_layers(init)
                    median = self._get_median(feature_layers)
                    return {"predictions": median}
        # generate quantile / anchor batch
        with timing_context(self, "forward.generate_anchors"):
            if self.training:
                if not self.fetch_quantile:
                    quantile_batch = None
                else:
                    quantile_batch = np.random.random([len(init), 1]).astype(np.float32)
                if not self.fetch_cdf:
                    anchor_batch = None
                else:
                    anchor_batch = np.random.random([len(init), 1]).astype(np.float32)
                    anchor_batch *= (self.y_max - self.y_min) + self.y_min
            else:
                if not self.fetch_cdf and not self.fetch_quantile:
                    quantile_batch = None
                else:
                    quantile_batch = self.default_anchors
                    n_repeat = int(len(init) / len(quantile_batch)) + 1
                    quantile_batch = np.repeat(quantile_batch, n_repeat)[: len(init)]
                if not self.fetch_cdf:
                    anchor_batch = None
                else:
                    anchor_batch = (
                        quantile_batch * (self.y_max - self.y_min) + self.y_min
                    )
            anchor_batch = self._convert_np_anchors(anchor_batch)
            quantile_batch = self._convert_np_anchors(quantile_batch)
        # build features
        with timing_context(self, "forward.build_features"):
            feature_layers = self._get_feature_layers(init)
        # build median
        with timing_context(self, "forward.median"):
            median = self._get_median(feature_layers)
        # construct basic info
        basic_info = BasicInfo(
            median, median.detach(), feature_layers, anchor_batch, quantile_batch
        )
        # is synthetic
        is_synthetic = kwargs.get("synthetic", False)
        # build distribution
        if not self.fetch_cdf:
            distribution_dict = {}
            pdf = sampled_pdf = None
        else:
            with timing_context(self, "forward.distribution"):
                fetch_anchors = self.training and not is_synthetic
                distribution_dict = self._get_distribution(basic_info, fetch_anchors)
            keys = ["pdf", "sampled_pdf"]
            pdf, sampled_pdf = map(distribution_dict.get, keys)
        # build quantile
        if not self.fetch_quantile:
            quantile_dict = {}
            quantile_residual = None
            quantile_residual_gradient = sampled_qr_gradient = None
        else:
            with timing_context(self, "forward.quantile"):
                quantile_dict = self._get_quantile_residual(
                    basic_info, self.training and not is_synthetic
                )
            # build quantile gradients
            sampled_qr_gradient = None
            quantile_residual, sampled_quantiles, sqr = map(
                quantile_dict.get,
                ["quantile_residual", "sampled_quantiles", "sampled_quantile_residual"],
            )
            fetch_quantile_gradient = self._use_gradient_loss and self.training
            if not is_synthetic and not fetch_quantile_gradient:
                quantile_residual_gradient = None
            else:
                assert quantile_batch is not None
                assert quantile_residual is not None
                with timing_context(self, "forward.quantile_gradient"):
                    quantile_residual_gradient = get_gradient(
                        quantile_residual,
                        quantile_batch,
                        True,
                        True,
                    )
                    if sqr is not None:
                        assert sampled_quantiles is not None
                        sampled_qr_gradient = get_gradient(
                            sqr,
                            sampled_quantiles,
                            True,
                            True,
                        )
        # build dual
        if (
            not self._joint_training
            or not self.fetch_cdf
            or not self.fetch_quantile
            or is_synthetic
        ):
            dual_cdf_dict, dual_quantile_dict = {}, {}
        else:
            assert quantile_residual is not None
            with timing_context(self, "forward.dual_cdf"):
                dual_cdf_dict = self._get_dual_cdf(basic_info, quantile_residual)
            with timing_context(self, "forward.dual_quantile"):
                dual_quantile_dict = self._get_dual_quantile(
                    basic_info, distribution_dict["cdf"]
                )
        # construct results
        rs = {
            "init": init,
            "predictions": median,
            "median_detach": basic_info.median_detach,
            "feature_layers": feature_layers,
            "anchor_batch": anchor_batch,
            "quantile_batch": quantile_batch,
            "pdf": pdf,
            "quantile_residual_gradient": quantile_residual_gradient,
            "sampled_pdf": sampled_pdf,
            "sampled_qr_gradient": sampled_qr_gradient,
        }
        for d in (distribution_dict, quantile_dict, dual_cdf_dict, dual_quantile_dict):
            rs.update(d)
        return rs

    # API

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        forward_dict = {}
        x_batch = batch["x_batch"]
        init = self._split_features(x_batch, batch_indices).merge()
        if self.tr_data.is_ts:
            init = init.view(init.shape[0], -1)
        predict_pdf, predict_cdf = map(kwargs.get, ["predict_pdf", "predict_cdf"])
        predict_quantile, predict_median_residual = map(
            kwargs.get, ["predict_quantile", "predict_median_residual"]
        )
        if predict_pdf or predict_cdf:
            y = kwargs.get("y")
            if y is None:
                raise ValueError(f"pdf / cdf cannot be predicted without y")
            y_batch = self._expand(len(init), y, numpy=True)
            y_batch = self.tr_data.transform_labels(y_batch)
            y_batch = to_torch(y_batch).to(self.device)
            if predict_pdf:
                forward_dict["pdf"] = self._predict_pdf(init, y_batch)
            if predict_cdf:
                forward_dict["cdf"] = self._predict_cdf(init, y_batch)
        if predict_quantile:
            q = kwargs.get("q")
            if q is None:
                raise ValueError(f"quantile cannot be predicted without q")
            q_batch = self._expand(len(init), q)
            forward_dict["quantile"] = self._predict_quantile(init, q_batch)
        if predict_median_residual:
            expand = lambda sign: self._expand(len(init), sign)
            sign_batch = torch.cat(list(map(expand, [1, -1])))
            init2 = init.repeat([2, 1])
            pos, neg = torch.chunk(
                self._predict_median_residual(init2, sign_batch), 2, dim=0
            )
            forward_dict["mr_pos"] = pos
            forward_dict["mr_neg"] = neg
        if not forward_dict:
            forward_dict = self._core(init, **shallow_copy_dict(kwargs))
        return forward_dict

    def loss_function(
        self,
        batch: tensor_dict_type,
        batch_indices: np.ndarray,
        forward_results: tensor_dict_type,
    ) -> Dict[str, torch.Tensor]:
        init = forward_results["init"]
        x_batch, y_batch = map(batch.get, ["x_batch", "y_batch"])
        losses, losses_dict = self.loss(forward_results, y_batch)
        if (
            self.training
            and self._synthetic_step > 0
            and self._step_count % self._synthetic_step == 0
        ):
            with timing_context(self, "synthetic.get_batch"):
                init_min = torch.min(init, dim=0)[0].view(*init.shape[1:])
                init_max = torch.max(init, dim=0)[0].view(*init.shape[1:])
                init_diff = init_max - init_min
                lower_bound = 0.5 * (self._synthetic_range - 1) * init_diff
                synthetic_init = init.new_empty(init.shape)
                synthetic_init.uniform_(0, 1)
                synthetic_init = synthetic_init * init_diff * self._synthetic_range
                synthetic_init = synthetic_init - (lower_bound - init_min)
            with timing_context(self, "synthetic.forward"):
                synthetic_outputs = self._core(
                    synthetic_init, no_loss=False, synthetic=True
                )
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
            if self._quantile_anchors is None:
                losses_dict["ddr"] = losses_dict["loss"]
            else:
                assert isinstance(y_batch, torch.Tensor)
                q_losses = []
                y_batch = to_numpy(y_batch)
                for q in self._quantile_anchors:
                    self.q_metric.config["q"] = q
                    yq = self._predict_quantile(init, self._expand(len(init), q))
                    q_losses.append(self.q_metric.metric(y_batch, to_numpy(yq)))
                quantile_metric = -sum(q_losses) / len(q_losses) * self.q_metric.sign
                ddr_loss = torch.tensor([quantile_metric], dtype=torch.float32)
                losses_dict["ddr"] = ddr_loss
        return losses_dict


__all__ = ["DDR"]
