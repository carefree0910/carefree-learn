import os
import json
import torch

import numpy as np

from typing import *
from functools import partial
from onnxruntime import InferenceSession
from tqdm.autonotebook import tqdm
from cftool.ml import Metrics
from cftool.misc import shallow_copy_dict
from cftool.misc import lock_manager
from cftool.misc import Saving
from cftool.misc import LoggingMixin
from cfdata.types import np_int_type

from .data import PrefetchLoader
from .types import data_type
from .types import np_dict_type
from .protocol import DataProtocol
from .protocol import SamplerProtocol
from .protocol import DataLoaderProtocol
from .misc.toolkit import to_prob
from .misc.toolkit import to_numpy
from .misc.toolkit import is_float
from .misc.toolkit import to_standard
from .misc.toolkit import collate_np_dicts
from .misc.toolkit import eval_context
from .models.base import ModelBase


class PreProcessor(LoggingMixin):
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
            self.input_names = sorted(self.input_sample.keys())
            self.output_names = sorted(outputs.keys())
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
        with eval_context(model):
            torch.onnx.export(
                model,
                self.input_sample,
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


class Inference(LoggingMixin):
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
        self._use_grad_in_predict = False
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
            self.output_probabilities = self.onnx.output_probabilities
            self.model = None
        else:
            self.onnx = None
            self.model = model
            if model is None:
                raise ValueError("either `onnx_config` or `model` should be provided")
            self.output_probabilities = model.output_probabilities

    def __str__(self) -> str:
        return f"Inference({self.model if self.model is not None else 'ONNX'})"

    __repr__ = __str__

    @property
    def binary_config(self) -> Dict[str, Any]:
        return {
            "binary_metric": self.binary_metric,
            "binary_threshold": self.binary_threshold,
        }

    @property
    def need_binary_threshold(self) -> bool:
        if not self.use_binary_threshold:
            return False
        return self.is_binary and self.binary_metric is not None

    def inject_binary_config(self, config: Dict[str, Any]) -> None:
        self.binary_metric = config.get("binary_metric")
        self.binary_threshold = config.get("binary_threshold")

    def to_tqdm(self, loader: PrefetchLoader) -> Union[tqdm, PrefetchLoader]:
        if not self.use_tqdm:
            return loader
        return tqdm(loader, total=len(loader), leave=False, position=2)

    def generate_binary_threshold(
        self,
        loader: Optional[PrefetchLoader] = None,
        loader_name: Optional[str] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.need_binary_threshold:
            return None
        if loader is None:
            raise ValueError("`loader` should be provided")
        results = self.predict(
            loader,
            return_all=True,
            returns_probabilities=True,
            loader_name=loader_name,
        )
        labels = results["labels"]
        probabilities = results["predictions"]
        try:
            threshold = Metrics.get_binary_threshold(
                labels,
                probabilities,
                self.binary_metric,
            )
            self.binary_threshold = threshold.item()
        except ValueError:
            self.binary_threshold = None

        if loader_name == "tr":
            return None
        return labels, probabilities

    def predict_with(self, probabilities: np.ndarray) -> np.ndarray:
        if not self.is_binary or self.binary_threshold is None:
            return probabilities.argmax(1).reshape([-1, 1])
        predictions = (
            (probabilities[..., 1] >= self.binary_threshold)
            .astype(np_int_type)
            .reshape([-1, 1])
        )
        return predictions

    def predict(
        self,
        loader: PrefetchLoader,
        *,
        return_all: bool = False,
        requires_recover: bool = True,
        returns_probabilities: bool = False,
        loader_name: Optional[str] = None,
        batch_step: int = -1,
        use_tqdm: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, np_dict_type]:

        # Notice : when `return_all` is True,
        #  there might not be `predictions` key in the results

        kwargs = shallow_copy_dict(kwargs)

        # calculate
        use_grad = kwargs.pop("use_grad", self._use_grad_in_predict)
        try:
            labels, results = self._get_results(
                use_grad,
                loader,
                loader_name,
                batch_step,
                use_tqdm,
                **shallow_copy_dict(kwargs),
            )
        except:
            use_grad = self._use_grad_in_predict = True
            labels, results = self._get_results(
                use_grad,
                loader,
                loader_name,
                batch_step,
                use_tqdm,
                **shallow_copy_dict(kwargs),
            )

        # collate
        collated = collate_np_dicts(results)
        if labels:
            labels = np.vstack(labels)
            collated["labels"] = labels

        # regression
        if self.data.is_reg:
            return_key = kwargs.get("return_key", "predictions")
            fn = partial(self.data.recover_labels, inplace=True)
            if not return_all:
                predictions = collated[return_key]
                if requires_recover:
                    if predictions.shape[1] == 1:
                        return fn(predictions)
                    return np.apply_along_axis(fn, axis=0, arr=predictions).squeeze()
                return predictions
            if not requires_recover:
                return collated
            recovered = {}
            for k, v in collated.items():
                if is_float(v):
                    if v.shape[1] == 1:
                        v = fn(v)
                    else:
                        v = np.apply_along_axis(fn, axis=0, arr=v).squeeze()
                recovered[k] = v
            return recovered

        # classification
        def _return(new_predictions: np.ndarray) -> Union[np.ndarray, np_dict_type]:
            if not return_all:
                return new_predictions
            collated["predictions"] = new_predictions
            return collated

        predictions = collated["logits"] = collated["predictions"]
        if returns_probabilities:
            if not self.output_probabilities:
                predictions = to_prob(predictions)
            return _return(predictions)
        if not self.is_binary or self.binary_threshold is None:
            return _return(predictions.argmax(1).reshape([-1, 1]))

        if self.output_probabilities:
            probabilities = predictions
        else:
            probabilities = to_prob(predictions)
        return _return(self.predict_with(probabilities))

    def _get_results(
        self,
        use_grad: bool,
        loader: PrefetchLoader,
        loader_name: Optional[str],
        batch_step: int,
        use_tqdm: bool,
        **kwargs: Any,
    ) -> Tuple[List[np.ndarray], List[np_dict_type]]:
        if use_tqdm:
            loader = self.to_tqdm(loader)
        results, labels = [], []
        for batch, batch_indices in loader:
            y_batch = batch["y_batch"]
            if y_batch is not None:
                if not isinstance(y_batch, np.ndarray):
                    y_batch = to_numpy(y_batch)
                labels.append(y_batch)
            if self.onnx is not None:
                rs = self.onnx.inference(batch)
            else:
                assert self.model is not None
                with eval_context(self.model, use_grad=use_grad):
                    rs = self.model(
                        batch,
                        batch_indices,
                        loader_name,
                        batch_step,
                        **shallow_copy_dict(kwargs),
                    )
                for k, v in rs.items():
                    if isinstance(v, torch.Tensor):
                        rs[k] = to_numpy(v)
            results.append(rs)
        return labels, results


__all__ = [
    "PreProcessor",
    "ONNX",
    "Inference",
]
