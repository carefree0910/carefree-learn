import os
import json
import onnx
import torch
import shutil

import numpy as np

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from onnxsim import simplify as onnx_simplify
from cftool.misc import shallow_copy_dict
from cftool.misc import lock_manager
from cftool.misc import Saving
from onnxsim.onnx_simplifier import get_input_names

from .trainer import make_trainer
from ...types import np_dict_type
from ...types import tensor_dict_type
from ...types import sample_weights_type
from ...types import states_callback_type
from ...trainer import get_sorted_checkpoints
from ...trainer import Trainer
from ...trainer import DeviceInfo
from ...protocol import ONNX
from ...protocol import LossProtocol
from ...protocol import ModelProtocol
from ...protocol import MetricsOutputs
from ...protocol import InferenceProtocol
from ...protocol import DataLoaderProtocol
from ...protocol import ModelWithCustomSteps
from ...constants import PT_PREFIX
from ...constants import INFO_PREFIX
from ...constants import SCORES_FILE
from ...constants import WARNING_PREFIX
from ...constants import PREDICTIONS_KEY
from ...constants import CHECKPOINTS_FOLDER
from ...constants import BATCH_INDICES_KEY
from ...misc.toolkit import to_numpy
from ...misc.toolkit import get_ddp_info
from ...misc.toolkit import fix_denormal_states
from ...misc.toolkit import prepare_workplace_from
from ...misc.toolkit import eval_context
from ...misc.toolkit import WithRegister
from ...misc.internal_ import DataModule
from ...misc.internal_ import DLDataModule
from ...misc.internal_.losses import multi_prefix_mapping
from ...misc.internal_.losses import AuxLoss


pipeline_dict: Dict[str, Type["PipelineProtocol"]] = {}
split_sw_type = Tuple[Optional[np.ndarray], Optional[np.ndarray]]


def _norm_sw(sample_weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if sample_weights is None:
        return None
    return sample_weights / sample_weights.sum()


def _split_sw(sample_weights: sample_weights_type) -> split_sw_type:
    if sample_weights is None:
        train_weights = valid_weights = None
    else:
        if not isinstance(sample_weights, np.ndarray):
            train_weights, valid_weights = sample_weights
        else:
            train_weights, valid_weights = sample_weights, None
    train_weights, valid_weights = map(_norm_sw, [train_weights, valid_weights])
    return train_weights, valid_weights


class PipelineProtocol(WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type["PipelineProtocol"]] = pipeline_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def build(self, data_info: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def fit(
        self,
        data: DataModule,
        *,
        sample_weights: sample_weights_type = None,
    ) -> "PipelineProtocol":
        pass

    @abstractmethod
    def predict(self, data: DataModule, **predict_kwargs: Any) -> np_dict_type:
        pass

    @abstractmethod
    def save(self, export_folder: str) -> "PipelineProtocol":
        pass

    @staticmethod
    @abstractmethod
    def load(export_folder: str) -> "PipelineProtocol":
        pass


class DLPipeline(PipelineProtocol, metaclass=ABCMeta):
    loss: LossProtocol
    model: ModelProtocol
    trainer: Trainer
    inference: InferenceProtocol
    inference_base: Type[InferenceProtocol]
    device_info: DeviceInfo

    pipeline_file: str = "pipeline.txt"
    config_file: str = "config.json"
    trainer_config_file: str = "trainer_config.json"
    data_info_name: str = "data_info"
    metrics_log_file: str = "metrics.txt"

    final_results_file = "final_results.json"
    config_bundle_name = "config_bundle"
    onnx_kwargs_file: str = "onnx.json"
    onnx_keys_file: str = "onnx_keys.json"

    config: Dict[str, Any]
    input_dim: Optional[int]

    def __init__(
        self,
        *,
        loss_name: str,
        loss_config: Optional[Dict[str, Any]] = None,
        # trainer
        state_config: Optional[Dict[str, Any]] = None,
        num_epoch: int = 40,
        max_epoch: int = 1000,
        fixed_epoch: Optional[int] = None,
        fixed_steps: Optional[int] = None,
        log_steps: Optional[int] = None,
        valid_portion: float = 1.0,
        amp: bool = False,
        clip_norm: float = 0.0,
        metric_names: Optional[Union[str, List[str]]] = None,
        metric_configs: Optional[Dict[str, Any]] = None,
        use_losses_as_metrics: Optional[bool] = None,
        loss_metrics_weights: Optional[Dict[str, float]] = None,
        recompute_train_losses_in_eval: bool = True,
        monitor_names: Optional[Union[str, List[str]]] = None,
        monitor_configs: Optional[Dict[str, Any]] = None,
        callback_names: Optional[Union[str, List[str]]] = None,
        callback_configs: Optional[Dict[str, Any]] = None,
        lr: Optional[float] = None,
        optimizer_name: Optional[str] = None,
        scheduler_name: Optional[str] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
        workplace: str = "_logs",
        finetune_config: Optional[Dict[str, Any]] = None,
        tqdm_settings: Optional[Dict[str, Any]] = None,
        # misc
        in_loading: bool = False,
    ):
        super().__init__()
        self.loss_name = loss_name
        self.loss_config = loss_config
        self.trainer_config: Dict[str, Any] = {
            "state_config": state_config,
            "num_epoch": num_epoch,
            "max_epoch": max_epoch,
            "fixed_epoch": fixed_epoch,
            "fixed_steps": fixed_steps,
            "log_steps": log_steps,
            "valid_portion": valid_portion,
            "amp": amp,
            "clip_norm": clip_norm,
            "metric_names": metric_names,
            "metric_configs": metric_configs,
            "use_losses_as_metrics": use_losses_as_metrics,
            "loss_metrics_weights": loss_metrics_weights,
            "recompute_train_losses_in_eval": recompute_train_losses_in_eval,
            "monitor_names": monitor_names,
            "monitor_configs": monitor_configs,
            "callback_names": callback_names,
            "callback_configs": callback_configs,
            "lr": lr,
            "optimizer_name": optimizer_name,
            "scheduler_name": scheduler_name,
            "optimizer_config": optimizer_config,
            "scheduler_config": scheduler_config,
            "optimizer_settings": optimizer_settings,
            "workplace": workplace,
            "finetune_config": finetune_config,
            "tqdm_settings": tqdm_settings,
        }
        self.in_loading = in_loading
        self.built = False
        self.device_info = DeviceInfo(None, None)

    # properties

    @property
    def device(self) -> torch.device:
        return self.device_info.device

    @property
    def is_rank_0(self) -> bool:
        ddp_info = get_ddp_info()
        if ddp_info is None:
            return True
        if ddp_info.rank == 0:
            return True
        return False

    # abstract

    @abstractmethod
    def _make_new_loader(
        self,
        data: DLDataModule,
        batch_size: int,
        **kwargs: Any,
    ) -> DataLoaderProtocol:
        pass

    @abstractmethod
    def _prepare_modules(self, data_info: Dict[str, Any]) -> None:
        pass

    # internal

    def _write_pipeline_info(self, folder: str) -> None:
        with open(os.path.join(folder, self.pipeline_file), "w") as f:
            f.write(self.__identifier__)

    def _prepare_workplace(self) -> None:
        if self.is_rank_0 and not self.in_loading:
            workplace = prepare_workplace_from(self.trainer_config["workplace"])
            self.trainer_config["workplace"] = workplace
            self.trainer_config["data_info_name"] = self.data_info_name
            self.trainer_config["metrics_log_file"] = self.metrics_log_file
            self._write_pipeline_info(workplace)
            with open(os.path.join(workplace, self.config_file), "w") as f:
                json.dump(self.config, f)

    def _prepare_loss(self) -> None:
        if self.in_loading:
            return None
        aux_loss_name = AuxLoss.parse(self.loss_name)
        if aux_loss_name is not None:
            self.loss_name = aux_loss_name
        for prefix, base in multi_prefix_mapping.items():
            if self.loss_name.startswith(f"{prefix}:"):
                loss_names = self.loss_name.split(":")[1].split(",")
                self.loss_name = base.register_(loss_names)
        self.loss = LossProtocol.make(self.loss_name, self.loss_config or {})

    def _prepare_trainer_defaults(self, data_info: Dict[str, Any]) -> None:
        # set some trainer defaults to deep learning tasks which work well in practice
        if self.trainer_config["monitor_names"] is None:
            self.trainer_config["monitor_names"] = ["mean_std", "plateau"]
        tqdm_settings = self.trainer_config["tqdm_settings"]
        callback_names = self.trainer_config["callback_names"]
        callback_configs = self.trainer_config["callback_configs"]
        optimizer_settings = self.trainer_config["optimizer_settings"]
        if callback_names is None:
            callback_names = []
        if callback_configs is None:
            callback_configs = {}
        if isinstance(callback_names, str):
            callback_names = [callback_names]
        auto_callback = self.trainer_config.get("auto_callback", True)
        if "mlflow" in callback_names and auto_callback:
            mlflow_config = callback_configs.setdefault("mlflow", {})
            mlflow_config.setdefault("experiment_name", self.model.__identifier__)
        if "_log_metrics_msg" not in callback_names and auto_callback:
            callback_names.insert(0, "_log_metrics_msg")
            verbose = False
            if tqdm_settings is None or not tqdm_settings.get("use_tqdm", False):
                verbose = True
            log_metrics_msg_config = callback_configs.setdefault("_log_metrics_msg", {})
            log_metrics_msg_config.setdefault("verbose", verbose)
        self.trainer_config["tqdm_settings"] = tqdm_settings
        self.trainer_config["callback_names"] = callback_names
        self.trainer_config["callback_configs"] = callback_configs
        self.trainer_config["optimizer_settings"] = optimizer_settings

    def _save_misc(self, export_folder: str) -> float:
        os.makedirs(export_folder, exist_ok=True)
        self._write_pipeline_info(export_folder)
        self.data.save(export_folder)
        # final results
        try:
            final_results = self.trainer.final_results
            if final_results is None:
                raise ValueError("`final_results` are not generated yet")
        except AttributeError as e:
            print(f"{WARNING_PREFIX}{e}, so `final_results` cannot be accessed")
            final_results = MetricsOutputs(0.0, {"unknown": 0.0})
        with open(os.path.join(export_folder, self.final_results_file), "w") as f:
            json.dump(final_results, f)
        # config bundle
        config_bundle = {
            "config": shallow_copy_dict(self.config),
            "device_info": self.device_info,
        }
        Saving.save_dict(config_bundle, self.config_bundle_name, export_folder)
        return final_results.final_score

    @classmethod
    def _load_infrastructure(
        cls,
        export_folder: str,
        cuda: Optional[str],
        pre_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        post_callback: Optional[Callable[["DLPipeline", Dict[str, Any]], None]] = None,
    ) -> "DLPipeline":
        config_bundle = Saving.load_dict(cls.config_bundle_name, export_folder)
        if pre_callback is not None:
            pre_callback(config_bundle)
        config = config_bundle["config"]
        config["in_loading"] = True
        m = cls(**config)
        device_info = DeviceInfo(*config_bundle["device_info"])
        device_info = device_info._replace(cuda=cuda)
        m.device_info = device_info
        if post_callback is not None:
            post_callback(m, config_bundle)
        return m

    @classmethod
    def _load_states_callback(cls, m: Any, states: Dict[str, Any]) -> Dict[str, Any]:
        return states

    @classmethod
    def _load_states_from(cls, m: Any, folder: str) -> Dict[str, Any]:
        checkpoints = get_sorted_checkpoints(folder)
        if not checkpoints:
            msg = f"{WARNING_PREFIX}no model file found in {folder}"
            raise ValueError(msg)
        checkpoint_path = os.path.join(folder, checkpoints[0])
        states = torch.load(checkpoint_path, map_location=m.device)
        return cls._load_states_callback(m, states)

    # core

    def _fit(self, data: DLDataModule, cuda: Optional[str]) -> None:
        self.data = data
        self.build(data.info)
        self.trainer.fit(
            data,
            self.loss,
            self.model,
            self.inference,
            config_export_file=self.trainer_config_file,
            cuda=cuda,
        )
        self.device_info = self.trainer.device_info

    # api

    def build(self, data_info: Dict[str, Any]) -> None:
        if self.built:
            return None
        self._prepare_modules(data_info)
        if self.in_loading:
            return None
        self._prepare_trainer_defaults(data_info)
        trainer_config = shallow_copy_dict(self.trainer_config)
        if isinstance(self.model, ModelWithCustomSteps):
            self.model.permute_trainer_config(trainer_config)
        self.trainer = make_trainer(**trainer_config)
        self.built = True

    def fit(  # type: ignore
        self,
        data: DLDataModule,
        *,
        sample_weights: sample_weights_type = None,
        cuda: Optional[Union[int, str]] = None,
    ) -> "DLPipeline":
        data.prepare(sample_weights)
        if cuda is not None:
            cuda = str(cuda)
        self._fit(data, cuda)
        return self

    def predict(  # type: ignore
        self,
        data: DLDataModule,
        *,
        batch_size: int = 128,
        make_loader_kwargs: Optional[Dict[str, Any]] = None,
        **predict_kwargs: Any,
    ) -> np_dict_type:
        loader = self._make_new_loader(data, batch_size, **(make_loader_kwargs or {}))
        predict_kwargs = shallow_copy_dict(predict_kwargs)
        outputs = self.inference.get_outputs(loader, **predict_kwargs)
        return outputs.forward_results

    def save(
        self,
        export_folder: str,
        *,
        compress: bool = True,
        remove_original: bool = True,
    ) -> "DLPipeline":
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            score = self._save_misc(export_folder)
            self.trainer.save_checkpoint(score, export_folder, no_history=True)
            if compress:
                Saving.compress(abs_folder, remove_original=remove_original)
        return self

    @classmethod
    def pack(
        cls,
        workplace: str,
        *,
        step: Optional[str] = None,
        config_bundle_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        pack_folder: Optional[str] = None,
        cuda: Optional[str] = None,
    ) -> str:
        if pack_folder is None:
            pack_name = f"packed{'' if step is None else f'_{step}'}"
            pack_folder = os.path.join(workplace, pack_name)
        if os.path.isdir(pack_folder):
            print(f"{WARNING_PREFIX}'{pack_folder}' already exists, it will be erased")
            shutil.rmtree(pack_folder)
        os.makedirs(pack_folder)
        abs_folder = os.path.abspath(pack_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [pack_folder]):
            shutil.copyfile(
                os.path.join(workplace, cls.pipeline_file),
                os.path.join(pack_folder, cls.pipeline_file),
            )
            checkpoint_folder = os.path.join(workplace, CHECKPOINTS_FOLDER)
            if step is not None:
                best_file = f"{PT_PREFIX}{step}.pt"
            else:
                best_file = get_sorted_checkpoints(checkpoint_folder)[0]
            new_file = f"{PT_PREFIX}-1.pt"
            shutil.copyfile(
                os.path.join(checkpoint_folder, best_file),
                os.path.join(pack_folder, new_file),
            )
            with open(os.path.join(checkpoint_folder, SCORES_FILE), "r") as rf:
                scores = json.load(rf)
            with open(os.path.join(pack_folder, SCORES_FILE), "w") as wf:
                json.dump({new_file: scores[best_file]}, wf)
            with open(os.path.join(workplace, cls.config_file), "r") as rf:
                config = json.load(rf)
            config_bundle = {
                "config": config,
                "device_info": DeviceInfo(cuda, None),
            }
            if config_bundle_callback is not None:
                config_bundle_callback(config_bundle)
            Saving.save_dict(config_bundle, cls.config_bundle_name, pack_folder)
            shutil.copytree(
                os.path.join(workplace, DataModule.package_folder),
                os.path.join(pack_folder, DataModule.package_folder),
            )
            Saving.compress(abs_folder, remove_original=True)
        return pack_folder

    @staticmethod
    def get_base(workplace: str) -> Type["DLPipeline"]:
        with open(os.path.join(workplace, DLPipeline.pipeline_file), "r") as f:
            return DLPipeline.get(f.read())

    @staticmethod
    def load(
        export_folder: str,
        *,
        cuda: Optional[Union[int, str]] = None,
        compress: bool = True,
        states_callback: states_callback_type = None,
        pre_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        post_callback: Optional[Callable[["DLPipeline", Dict[str, Any]], None]] = None,
    ) -> "DLPipeline":
        if export_folder.endswith(".zip"):
            export_folder = export_folder[:-4]
        base_folder = os.path.dirname(os.path.abspath(export_folder))
        with lock_manager(base_folder, [export_folder]):
            with Saving.compress_loader(export_folder, compress):
                cls = DLPipeline.get_base(export_folder)
                m = cls._load_infrastructure(
                    export_folder,
                    None if cuda is None else str(cuda),
                    pre_callback,
                    post_callback,
                )
                data_info = DataModule.load(export_folder)
                m._prepare_modules(data_info)
                m.model.to(m.device)
                # restore checkpoint
                states = cls._load_states_from(m, export_folder)
                if states_callback is not None:
                    states = states_callback(m, states)
                m.model.load_state_dict(states)
        return m

    def to_onnx(
        self,
        export_folder: str,
        dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
        *,
        onnx_file: str = "model.onnx",
        opset: int = 11,
        simplify: bool = True,
        onnx_only: bool = False,
        input_sample: Optional[tensor_dict_type] = None,
        num_samples: Optional[int] = None,
        compress: Optional[bool] = None,
        remove_original: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "DLPipeline":
        # prepare
        model = self.model.cpu()
        if input_sample is None:
            if getattr(self, "trainer", None) is None:
                msg = "either `input_sample` or `trainer` should be provided"
                raise ValueError(msg)
            input_sample = self.trainer.input_sample
            input_sample.pop(BATCH_INDICES_KEY)
        assert isinstance(input_sample, dict)
        if num_samples is not None:
            input_sample = {k: v[:num_samples] for k, v in input_sample.items()}
        with eval_context(model):
            forward_results = model.onnx_forward(shallow_copy_dict(input_sample))
        if not isinstance(forward_results, dict):
            forward_results = {PREDICTIONS_KEY: forward_results}
        input_names = sorted(input_sample.keys())
        # setup
        kwargs = shallow_copy_dict(kwargs)
        kwargs["input_names"] = input_names
        kwargs["output_names"] = sorted(forward_results.keys())
        kwargs["opset_version"] = opset
        kwargs["export_params"] = True
        kwargs["do_constant_folding"] = True
        if dynamic_axes is None:
            dynamic_axes = {}
        elif isinstance(dynamic_axes, list):
            dynamic_axes = {axis: f"axis.{axis}" for axis in dynamic_axes}
        if num_samples is None:
            dynamic_axes[0] = "batch_size"
        dynamic_axes_settings = {}
        for name in input_names + kwargs["output_names"]:
            dynamic_axes_settings[name] = dynamic_axes
        kwargs["dynamic_axes"] = dynamic_axes_settings
        kwargs["verbose"] = verbose
        # export
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)

        class ONNXWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(self, batch: Dict[str, Any]) -> Any:
                return self.model.onnx_forward(batch)

        with lock_manager(base_folder, []) as lock:
            onnx_path = os.path.join(export_folder, onnx_file)
            if onnx_only:
                lock._stuffs = [onnx_path]
                os.makedirs(export_folder, exist_ok=True)
            else:
                lock._stuffs = [export_folder]
                self._save_misc(export_folder)
                with open(os.path.join(export_folder, self.onnx_kwargs_file), "w") as f:
                    json.dump(kwargs, f)
            m_onnx = ONNXWrapper()
            input_keys = sorted(input_sample)
            original_states = model.state_dict()
            fixed_states = fix_denormal_states(original_states, verbose=verbose)
            with eval_context(m_onnx):
                model.load_state_dict(fixed_states)
                torch.onnx.export(
                    m_onnx,
                    (input_sample, {}),
                    onnx_path,
                    **shallow_copy_dict(kwargs),
                )
                model.load_state_dict(original_states)
                onnx_model = onnx.load(onnx_path)
                input_names = get_input_names(onnx_model)
                np_sample = {
                    name: to_numpy(tensor)
                    for name, tensor in input_sample.items()
                    if name in input_names
                }
                try:
                    if not simplify:
                        model_simplified = onnx_model
                        check = True
                    else:
                        model_simplified, check = onnx_simplify(
                            onnx_model,
                            input_data=np_sample,
                            dynamic_input_shape=bool(dynamic_axes),
                        )
                except Exception as err:
                    if verbose:
                        print(f"{WARNING_PREFIX}Failed to simplify ONNX model ({err})")
                    check = False
                if not check and verbose:
                    print(f"{INFO_PREFIX}Simplified ONNX model is not validated!")
                elif check and verbose and simplify:
                    print(f"{INFO_PREFIX}Simplified ONNX model is validated!")
                    onnx_model = model_simplified
                onnx.save(onnx_model, onnx_path)
                output_keys = sorted(m_onnx(input_sample))
            if not onnx_only:
                with open(os.path.join(export_folder, self.onnx_keys_file), "w") as f:
                    json.dump({"input": input_keys, "output": output_keys}, f)
            if compress or (compress is None and not onnx_only):
                Saving.compress(abs_folder, remove_original=remove_original)
        self.model.to(self.device)
        return self

    @classmethod
    def pack_onnx(
        cls,
        workplace: str,
        export_folder: str,
        dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
        *,
        step: Optional[str] = None,
        config_bundle_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        pack_folder: Optional[str] = None,
        states_callback: states_callback_type = None,
        pre_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        post_callback: Optional[Callable[["DLPipeline", Dict[str, Any]], None]] = None,
        onnx_file: str = "model.onnx",
        opset: int = 11,
        simplify: bool = True,
        onnx_only: bool = False,
        input_sample: Optional[tensor_dict_type] = None,
        num_samples: Optional[int] = None,
        compress: Optional[bool] = None,
        remove_original: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "DLPipeline":
        packed = cls.pack(
            workplace,
            step=step,
            config_bundle_callback=config_bundle_callback,
            pack_folder=pack_folder,
        )
        m = cls.load(
            packed,
            states_callback=states_callback,
            pre_callback=pre_callback,
            post_callback=post_callback,
        )
        m.to_onnx(
            export_folder,
            dynamic_axes,
            onnx_file=onnx_file,
            opset=opset,
            simplify=simplify,
            onnx_only=onnx_only,
            input_sample=input_sample,
            num_samples=num_samples,
            compress=compress,
            remove_original=remove_original,
            verbose=verbose,
            **kwargs,
        )
        return m

    @classmethod
    def from_onnx(
        cls,
        export_folder: str,
        *,
        onnx_file: str = "model.onnx",
        compress: bool = True,
        pre_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        post_callback: Optional[Callable[["DLPipeline", Dict[str, Any]], None]] = None,
    ) -> "DLPipeline":
        base_folder = os.path.dirname(os.path.abspath(export_folder))
        with lock_manager(base_folder, [export_folder]):
            with Saving.compress_loader(export_folder, compress):
                m = cls._load_infrastructure(
                    export_folder,
                    None,
                    pre_callback,
                    post_callback,
                )
                with open(os.path.join(export_folder, cls.onnx_kwargs_file), "r") as f:
                    onnx_kwargs = json.load(f)
                m_onnx = ONNX(
                    onnx_path=os.path.join(export_folder, onnx_file),
                    output_names=onnx_kwargs["output_names"],
                )
                m.inference = cls.inference_base(onnx=m_onnx)
        return m


__all__ = [
    "PipelineProtocol",
    "DLPipeline",
]
