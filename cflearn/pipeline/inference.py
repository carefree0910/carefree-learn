import torch

import numpy as np

from typing import *
from onnxruntime import InferenceSession

from .core import Pipeline
from ..types import np_dict_type
from ..types import tensor_dict_type
from ..misc.toolkit import to_torch
from ..misc.toolkit import to_standard
from ..misc.toolkit import eval_context


class ONNX:
    def __init__(
        self,
        pipeline: Pipeline,
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
        self.pipeline.trainer.onnx = self
        del self.pipeline.model, self.pipeline.trainer.model
        return self

    def inference(self, new_inputs: np_dict_type) -> np_dict_type:
        assert self.ort_session is not None
        ort_inputs = {
            node.name: to_standard(new_inputs[node.name])
            for node in self.ort_session.get_inputs()
        }
        return dict(zip(self.output_names, self.ort_session.run(None, ort_inputs)))


__all__ = [
    "ONNX",
]
