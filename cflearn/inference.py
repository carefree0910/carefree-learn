import os
import json
import torch

import numpy as np

from typing import *
from onnxruntime import InferenceSession
from cftool.misc import shallow_copy_dict
from cftool.misc import lock_manager
from cftool.misc import Saving

from .types import data_type
from .types import np_dict_type
from .protocol import DataProtocol
from .protocol import PrefetchLoader
from .protocol import SamplerProtocol
from .protocol import InferenceProtocol
from .protocol import DataLoaderProtocol
from .misc.toolkit import to_standard
from .misc.toolkit import eval_context
from .misc.toolkit import LoggingMixinWithRank
from .models.base import ModelBase


class PreProcessor(LoggingMixinWithRank):
    data_folder = "data"
    protocols_file = "protocols.json"
    sampler_config_name = "sampler_config"

    def __init__(
        self,
        data: DataProtocol,
        loader_protocol: str,
        sampler_protocol: str,
        sampler_config: Dict[str, Any],
    ):
        self.data = data
        self.data_protocol = data.__identifier__  # type: ignore
        self.loader_protocol = loader_protocol
        self.sampler_protocol = sampler_protocol
        self.sampler_config = sampler_config

    def make_sampler(
        self,
        data: DataProtocol,
        shuffle: bool,
        sample_weights: Optional[np.ndarray] = None,
    ) -> SamplerProtocol:
        config = shallow_copy_dict(self.sampler_config)
        config["shuffle"] = shuffle
        config["sample_weights"] = sample_weights
        return SamplerProtocol.make(self.sampler_protocol, data, **config)

    def make_inference_loader(
        self,
        x: data_type,
        device: Union[str, torch.device],
        batch_size: int = 256,
        *,
        is_onnx: bool,
        contains_labels: bool = False,
    ) -> PrefetchLoader:
        data = self.data.copy_to(x, None, contains_labels=contains_labels)
        loader = DataLoaderProtocol.make(
            self.loader_protocol,
            batch_size,
            self.make_sampler(data, False),
        )
        return PrefetchLoader(loader, device, is_onnx=is_onnx)

    def save(
        self,
        export_folder: str,
        *,
        compress: bool = True,
        save_data: bool = True,
        retain_data: bool = False,
        remove_original: bool = True,
    ) -> "PreProcessor":
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            Saving.prepare_folder(self, export_folder)
            if save_data:
                data_folder = os.path.join(export_folder, self.data_folder)
                self.data.save(
                    data_folder,
                    retain_data=retain_data,
                    compress=False,
                )
            with open(os.path.join(export_folder, self.protocols_file), "w") as f:
                json.dump(
                    {
                        "data": self.data_protocol,
                        "loader": self.loader_protocol,
                        "sampler": self.sampler_protocol,
                    },
                    f,
                )
            Saving.save_dict(
                self.sampler_config,
                self.sampler_config_name,
                export_folder,
            )
            if compress:
                Saving.compress(abs_folder, remove_original=remove_original)
        return self

    @classmethod
    def load(
        cls,
        export_folder: str,
        *,
        data: Optional[DataProtocol] = None,
        compress: bool = True,
    ) -> "PreProcessor":
        base_folder = os.path.dirname(os.path.abspath(export_folder))
        with lock_manager(base_folder, [export_folder]):
            with Saving.compress_loader(
                export_folder,
                compress,
                remove_extracted=True,
            ):
                with open(os.path.join(export_folder, cls.protocols_file), "r") as f:
                    protocols = json.load(f)
                data_protocol = protocols["data"]
                loader_protocol = protocols["loader"]
                sampler_protocol = protocols["sampler"]
                if data is None:
                    data_folder = os.path.join(export_folder, cls.data_folder)
                    data_base = DataProtocol.get(data_protocol)
                    data = data_base.load(data_folder, compress=False)
                sampler_cfg = Saving.load_dict(cls.sampler_config_name, export_folder)
        return cls(data, loader_protocol, sampler_protocol, sampler_cfg)


class ONNX:
    def __init__(
        self,
        *,
        model: Optional[ModelBase] = None,
        onnx_config: Optional[Dict[str, Any]] = None,
    ):
        if model is None and onnx_config is None:
            raise ValueError("either `model` or `onnx_config` should be provided")

        self.ort_session: Optional[InferenceSession]
        if onnx_config is not None:
            self.model = None
            onnx_path = onnx_config["onnx_path"]
            self.output_names = onnx_config["output_names"]
            self.output_probabilities = onnx_config["output_probabilities"]
            self.ort_session = InferenceSession(onnx_path)
        else:
            assert model is not None
            self.model = model.cpu()
            device, self.model.device = self.model.device, torch.device("cpu")
            self.ort_session = None
            self.input_sample = self.model.input_sample
            with eval_context(self.model):
                outputs = self.model(self.input_sample)
            self.input_names = list(self.input_sample.keys())
            self.output_names = list(outputs.keys())
            self.output_probabilities = model.output_probabilities
            self.model.device = device
            self.model.to(device)

    def to_onnx(
        self,
        onnx_path: str,
        dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
        *,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "ONNX":
        if self.model is None:
            raise ValueError("`model` is not provided")
        kwargs = shallow_copy_dict(kwargs)
        kwargs["input_names"] = self.input_names
        kwargs["output_names"] = self.output_names
        kwargs["opset_version"] = 11
        kwargs["export_params"] = True
        kwargs["do_constant_folding"] = True
        if dynamic_axes is None:
            dynamic_axes = {}
        elif isinstance(dynamic_axes, list):
            dynamic_axes = {axis: f"axis.{axis}" for axis in dynamic_axes}
        dynamic_axes[0] = "batch_size"
        dynamic_axes_settings = {}
        for name in self.input_names + self.output_names:
            dynamic_axes_settings[name] = dynamic_axes
        kwargs["dynamic_axes"] = dynamic_axes_settings
        kwargs["verbose"] = verbose
        model = self.model.cpu()

        class ONNXWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(self, batch: Dict[str, Any]) -> Any:
                return self.model(batch)

        onnx = ONNXWrapper()
        with eval_context(onnx):
            torch.onnx.export(
                onnx,
                (self.input_sample, {}),
                onnx_path,
                **shallow_copy_dict(kwargs),
            )
        model.to(model.device)
        return self

    def inference(self, new_inputs: np_dict_type) -> np_dict_type:
        if self.ort_session is None:
            raise ValueError("`onnx_path` is not provided")
        ort_inputs = {
            node.name: to_standard(new_inputs[node.name])
            for node in self.ort_session.get_inputs()
        }
        return dict(zip(self.output_names, self.ort_session.run(None, ort_inputs)))


class Inference(InferenceProtocol, LoggingMixinWithRank):
    def __init__(
        self,
        preprocessor: PreProcessor,
        *,
        model: Optional[ModelBase] = None,
        binary_config: Optional[Dict[str, Any]] = None,
        onnx_config: Optional[Dict[str, Any]] = None,
        use_binary_threshold: bool = True,
        use_tqdm: bool = True,
    ):
        if model is None and onnx_config is None:
            raise ValueError("either `model` or `onnx_config` should be provided")

        self.use_tqdm = use_tqdm
        self.data = preprocessor.data
        self.preprocessor = preprocessor
        self.use_binary_threshold = use_binary_threshold

        # binary case
        self.is_binary = self.data.num_classes == 2
        if binary_config is None:
            binary_config = {}
        self.inject_binary_config(binary_config)

        # onnx
        self.onnx: Optional[ONNX]
        self.model: Optional[ModelBase]

        if onnx_config is not None:
            if model is not None:
                self.log_msg(
                    "`model` and `onnx_config` are both provided, "
                    "`model` will be ignored"
                )
            self.onnx = ONNX(onnx_config=onnx_config)
            self.model = None
        else:
            self.onnx = None
            self.model = model
            if model is None:
                raise ValueError("either `onnx_config` or `model` should be provided")

    def __str__(self) -> str:
        return f"Inference({self.model if self.model is not None else 'ONNX'})"

    __repr__ = __str__

    def inject_binary_config(self, config: Dict[str, Any]) -> None:
        self.binary_metric = config.get("binary_metric")
        self.binary_threshold = config.get("binary_threshold")


__all__ = [
    "PreProcessor",
    "ONNX",
    "Inference",
]
