import os
import json
import torch

from abc import abstractmethod
from abc import ABC
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Callable
from typing import Optional
from cftool.misc import register_core
from cftool.misc import shallow_copy_dict
from cftool.misc import lock_manager
from cftool.misc import Saving

from .trainer import make_trainer
from ...types import data_type
from ...types import np_dict_type
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
from ...constants import SCORES_FILE
from ...constants import WARNING_PREFIX
from ...misc.toolkit import eval_context


pipeline_dict: Dict[str, Type["PipelineProtocol"]] = {}


class PipelineProtocol(ABC):
    loss: LossProtocol
    model: ModelProtocol
    trainer: Trainer
    inference: InferenceProtocol
    inference_base: Type[InferenceProtocol]
    device_info: DeviceInfo

    train_loader: DataLoaderProtocol
    valid_loader: Optional[DataLoaderProtocol]

    configs_file: str = "configs.json"
    metrics_log_file: str = "metrics.txt"

    data_folder: str = "data"
    final_results_file = "final_results.json"
    config_bundle_name = "config_bundle"
    onnx_file: str = "model.onnx"
    onnx_kwargs_file: str = "onnx.json"

    def __init__(
        self,
        *,
        loss_name: str,
        loss_config: Optional[Dict[str, Any]] = None,
        # valid split
        valid_split: Optional[Union[int, float]] = None,
        min_valid_split: int = 100,
        max_valid_split: int = 10000,
        max_valid_split_ratio: float = 0.5,
        valid_split_order: str = "auto",
        # data loader
        num_history: int = 1,
        shuffle_train: bool = True,
        shuffle_valid: bool = False,
        batch_size: int = 128,
        valid_batch_size: int = 512,
        # trainer
        state_config: Optional[Dict[str, Any]] = None,
        num_epoch: int = 40,
        max_epoch: int = 1000,
        valid_portion: float = 1.0,
        amp: bool = False,
        clip_norm: float = 0.0,
        metric_names: Optional[Union[str, List[str]]] = None,
        metric_configs: Optional[Dict[str, Any]] = None,
        monitor_names: Optional[Union[str, List[str]]] = None,
        monitor_configs: Optional[Dict[str, Any]] = None,
        callback_names: Optional[Union[str, List[str]]] = None,
        callback_configs: Optional[Dict[str, Any]] = None,
        optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
        workplace: str = "_logs",
        rank: Optional[int] = None,
        tqdm_settings: Optional[Dict[str, Any]] = None,
        # misc
        in_loading: bool = False,
    ):
        self.loss_name = loss_name
        self.loss_config = loss_config or {}
        self.valid_split = valid_split
        self.min_valid_split = min_valid_split
        self.max_valid_split = max_valid_split
        self.max_valid_split_ratio = max_valid_split_ratio
        self.valid_split_order = valid_split_order
        self.num_history = num_history
        self.shuffle_train = shuffle_train
        self.shuffle_valid = shuffle_valid
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.trainer_config: Dict[str, Any] = {
            "state_config": state_config,
            "num_epoch": num_epoch,
            "max_epoch": max_epoch,
            "valid_portion": valid_portion,
            "amp": amp,
            "clip_norm": clip_norm,
            "metric_names": metric_names,
            "metric_configs": metric_configs,
            "monitor_names": monitor_names,
            "monitor_configs": monitor_configs,
            "callback_names": callback_names,
            "callback_configs": callback_configs,
            "optimizer_settings": optimizer_settings,
            "workplace": workplace,
            "rank": rank,
            "tqdm_settings": tqdm_settings,
        }
        self.in_loading = in_loading

    @property
    def device(self) -> torch.device:
        return self.device_info.device

    @property
    def is_rank_0(self) -> bool:
        return self.trainer.is_rank_0

    def fit(
        self,
        x: data_type,
        y: data_type = None,
        x_valid: data_type = None,
        y_valid: data_type = None,
        *,
        cuda: Optional[str] = None,
    ) -> "PipelineProtocol":
        self._before_loop(x, y, x_valid, y_valid, cuda)
        self.trainer = make_trainer(**shallow_copy_dict(self.trainer_config))
        self.trainer.fit(
            self.loss,
            self.model,
            self.inference,
            self.train_loader,
            self.valid_loader,
            cuda=cuda,
        )
        self.device_info = self.trainer.device_info
        return self

    @abstractmethod
    def _before_loop(
        self,
        x: data_type,
        y: data_type = None,
        x_valid: data_type = None,
        y_valid: data_type = None,
        cuda: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def _make_new_loader(
        self,
        x: data_type,
        batch_size: int,
        **kwargs: Any,
    ) -> DataLoaderProtocol:
        pass

    def predict(
        self,
        x: data_type,
        *,
        batch_size: int = 128,
        make_loader_kwargs: Optional[Dict[str, Any]] = None,
        **predict_kwargs: Any,
    ) -> np_dict_type:
        loader = self._make_new_loader(x, batch_size, **(make_loader_kwargs or {}))
        predict_kwargs = shallow_copy_dict(predict_kwargs)
        if self.inference.onnx is None:
            predict_kwargs["device"] = self.device
        outputs = self.inference.get_outputs(loader, **predict_kwargs)
        return outputs.forward_results

    @abstractmethod
    def save(
        self,
        export_folder: str,
        *,
        compress: bool = True,
        retain_data: bool = False,
        remove_original: bool = True,
    ) -> "PipelineProtocol":
        pass

    @classmethod
    @abstractmethod
    def load(
        cls,
        export_folder: str,
        *,
        compress: bool = True,
        states_callback: states_callback_type = None,
    ) -> "PipelineProtocol":
        pass

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global pipeline_dict
        return register_core(name, pipeline_dict)


class DLPipeline(PipelineProtocol, metaclass=ABCMeta):
    config: Dict[str, Any]
    input_dim: Optional[int]

    @abstractmethod
    def _prepare_modules(self) -> None:
        pass

    def _save_misc(self, export_folder: str, retain_data: bool) -> float:
        os.makedirs(export_folder, exist_ok=True)
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
            "input_dim": self.input_dim,
        }
        Saving.save_dict(config_bundle, self.config_bundle_name, export_folder)
        return final_results.final_score

    def save(
        self,
        export_folder: str,
        *,
        compress: bool = True,
        retain_data: bool = False,
        remove_original: bool = True,
    ) -> "DLPipeline":
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            score = self._save_misc(export_folder, retain_data)
            self.trainer.save_checkpoint(score, export_folder)
            if compress:
                Saving.compress(abs_folder, remove_original=remove_original)
        return self

    @classmethod
    def _load_infrastructure(
        cls,
        export_folder: str,
        cuda: Optional[str],
    ) -> "DLPipeline":
        config_bundle = Saving.load_dict(cls.config_bundle_name, export_folder)
        config = config_bundle["config"]
        config["in_loading"] = True
        m = cls(**config)
        device_info = DeviceInfo(*config_bundle["device_info"])
        device_info = device_info._replace(cuda=cuda)
        m.device_info = device_info
        m.input_dim = config_bundle["input_dim"]
        return m

    @classmethod
    def load(
        cls,
        export_folder: str,
        *,
        cuda: Optional[str] = None,
        compress: bool = True,
        states_callback: states_callback_type = None,
    ) -> "DLPipeline":
        base_folder = os.path.dirname(os.path.abspath(export_folder))
        with lock_manager(base_folder, [export_folder]):
            with Saving.compress_loader(export_folder, compress):
                m = cls._load_infrastructure(export_folder, cuda)
                m._prepare_modules()
                m.model.to(m.device)
                # restore checkpoint
                score_path = os.path.join(export_folder, SCORES_FILE)
                checkpoints = get_sorted_checkpoints(score_path)
                if not checkpoints:
                    msg = f"{WARNING_PREFIX}no model file found in {export_folder}"
                    raise ValueError(msg)
                checkpoint_path = os.path.join(export_folder, checkpoints[0])
                states = torch.load(checkpoint_path, map_location=m.device)
                if states_callback is not None:
                    states = states_callback(m, states)
                m.model.load_state_dict(states)
        return m

    def to_onnx(
        self,
        export_folder: str,
        dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
        *,
        compress: bool = True,
        remove_original: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "DLPipeline":
        # prepare
        model = self.model.cpu()
        input_sample = self.trainer.input_sample
        input_sample.pop("batch_indices")
        with eval_context(model):
            forward_results = model(0, shallow_copy_dict(input_sample))
        input_names = sorted(input_sample.keys())
        output_names = sorted(forward_results.keys())
        # setup
        kwargs = shallow_copy_dict(kwargs)
        kwargs["input_names"] = input_names
        kwargs["output_names"] = output_names
        kwargs["opset_version"] = 11
        kwargs["export_params"] = True
        kwargs["do_constant_folding"] = True
        if dynamic_axes is None:
            dynamic_axes = {}
        elif isinstance(dynamic_axes, list):
            dynamic_axes = {axis: f"axis.{axis}" for axis in dynamic_axes}
        dynamic_axes[0] = "batch_size"
        dynamic_axes_settings = {}
        for name in input_names + output_names:
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
                return self.model(0, batch)

        with lock_manager(base_folder, [export_folder]):
            self._save_misc(export_folder, False)
            with open(os.path.join(export_folder, self.onnx_kwargs_file), "w") as f:
                json.dump(kwargs, f)
            onnx = ONNXWrapper()
            onnx_path = os.path.join(export_folder, self.onnx_file)
            with eval_context(onnx):
                torch.onnx.export(
                    onnx,
                    (input_sample, {}),
                    onnx_path,
                    **shallow_copy_dict(kwargs),
                )
            if compress:
                Saving.compress(abs_folder, remove_original=remove_original)
        self.model.to(self.device)
        return self

    @classmethod
    def from_onnx(
        cls,
        export_folder: str,
        *,
        compress: bool = True,
    ) -> "DLPipeline":
        base_folder = os.path.dirname(os.path.abspath(export_folder))
        with lock_manager(base_folder, [export_folder]):
            with Saving.compress_loader(export_folder, compress):
                m = cls._load_infrastructure(export_folder)
                with open(os.path.join(export_folder, cls.onnx_kwargs_file), "r") as f:
                    onnx_kwargs = json.load(f)
                onnx = ONNX(
                    onnx_path=os.path.join(export_folder, cls.onnx_file),
                    output_names=onnx_kwargs["output_names"],
                )
                m.inference = cls.inference_base(onnx=onnx)
        return m


__all__ = [
    "PipelineProtocol",
    "DLPipeline",
]
