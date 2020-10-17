import torch

import numpy as np

from typing import *
from tqdm import tqdm
from cftool.misc import LoggingMixin
from cfdata.types import np_int_type
from cfdata.types import np_float_type
from cfdata.tabular import DataLoader
from onnxruntime import InferenceSession

from ..types import np_dict_type
from ..types import tensor_dict_type
from ..models.base import ModelBase
from ..misc.toolkit import to_numpy
from ..misc.toolkit import to_torch
from ..misc.toolkit import to_standard
from ..misc.toolkit import eval_context
from ..misc.toolkit import collate_np_dicts


class ONNX:
    def __init__(
        self,
        pipeline: Any,
        onnx_path: str = None,
    ):
        self.pipeline = pipeline
        self.model = pipeline.model.cpu()
        self.ort_session: Optional[InferenceSession] = None
        if onnx_path is not None:
            self._init_onnx_session(onnx_path)
        # initialize
        self.input_sample = self.model.input_sample
        with eval_context(self.model):
            outputs = self.model(self.input_sample)
        self.input_names = sorted(self.input_sample.keys())
        self.output_names = sorted(outputs.keys())
        self.model.to(self.model.device)

    def _init_onnx_session(self, onnx_path: str) -> "ONNX":
        self.ort_session = InferenceSession(onnx_path)
        return self

    def to_onnx(
        self,
        onnx_path: str,
        dynamic_axes: Union[List[int], Dict[int, str]] = None,
        **kwargs: Any,
    ) -> "ONNX":
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
        model = self.model.cpu()
        with eval_context(model):
            torch.onnx.export(model, self.input_sample, onnx_path, **kwargs)
        self._init_onnx_session(onnx_path)
        model.to(model.device)
        return self

    def inject_onnx(self) -> "ONNX":
        self.pipeline.trainer.inference.onnx = self
        del self.pipeline.model, self.pipeline.trainer.model
        return self

    def inference(self, new_inputs: np_dict_type) -> np_dict_type:
        assert self.ort_session is not None
        ort_inputs = {
            node.name: to_standard(new_inputs[node.name])
            for node in self.ort_session.get_inputs()
        }
        return dict(zip(self.output_names, self.ort_session.run(None, ort_inputs)))


class Inference(LoggingMixin):
    def __init__(
        self,
        is_clf: bool,
        device: torch.device,
        *,
        model: Optional[ModelBase] = None,
        onnx_path: Optional[str] = None,
        use_tqdm: bool = True,
    ):
        self.use_tqdm = use_tqdm
        self.is_clf, self.device = is_clf, device
        self._use_grad_in_predict = False

        if model is None and onnx_path is None:
            raise ValueError("either `model` or `onnx_path` should be provided")
        if onnx_path is not None:
            if model is not None:
                self.log_msg(
                    "`model` and `onnx_path` are both provided, "
                    "`model` will be ignored"
                )
            self.onnx: Optional[ONNX] = None
            self.model = None
        else:
            self.onnx = None
            self.model = model

    def _to_device(self, arr: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        if arr is None:
            return arr
        return to_torch(arr).to(self.device)

    def to_tqdm(self, loader: DataLoader) -> Union[tqdm, DataLoader]:
        if not self.use_tqdm:
            return loader
        return tqdm(loader, total=len(loader), leave=False, position=2)

    def collate_batch(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> Union[np_dict_type, tensor_dict_type]:
        x_batch = x_batch.astype(np_float_type)
        if self.onnx is not None:
            if y_batch is None:
                y_batch = np.zeros([*x_batch.shape[:-1], 1], np_int_type)
            arrays = [x_batch, y_batch]
        else:
            x_batch, y_batch = list(map(self._to_device, [x_batch, y_batch]))
            if y_batch is not None and self.is_clf:
                y_batch = y_batch.to(torch.int64)
            arrays = [x_batch, y_batch]
        return dict(zip(["x_batch", "y_batch"], arrays))

    def predict(self, loader: DataLoader, **kwargs: Any) -> np_dict_type:
        use_grad = kwargs.pop("use_grad", self._use_grad_in_predict)
        try:
            labels, results = self._get_results(use_grad, loader, **kwargs)
        except:
            use_grad = self._use_grad_in_predict = True
            labels, results = self._get_results(use_grad, loader, **kwargs)
        collated = collate_np_dicts(results)
        if labels:
            labels = np.vstack(labels)
            collated["labels"] = labels
        return collated

    def _get_results(
        self,
        use_grad: bool,
        loader: DataLoader,
        **kwargs: Any,
    ) -> Tuple[List[np.ndarray], List[np_dict_type]]:
        return_indices, loader = loader.return_indices, self.to_tqdm(loader)
        results, labels = [], []
        for a, b in loader:
            if return_indices:
                x_batch, y_batch = a
            else:
                x_batch, y_batch = a, b
            if y_batch is not None:
                labels.append(y_batch)
            batch = self.collate_batch(x_batch, y_batch)
            if self.onnx is not None:
                rs = self.onnx.inference(batch)
            else:
                assert self.model is not None
                with eval_context(self.model, use_grad=use_grad):
                    rs = self.model(batch, **kwargs)
                for k, v in rs.items():
                    if isinstance(v, torch.Tensor):
                        rs[k] = to_numpy(v)
            results.append(rs)
        return labels, results


__all__ = [
    "ONNX",
    "Inference",
]
