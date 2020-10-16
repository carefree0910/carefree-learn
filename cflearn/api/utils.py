import torch

import numpy as np

from typing import *
from cftool.misc import update_dict
from onnxruntime import InferenceSession

from ..bases import *
from ..misc.toolkit import *
from .basic import make


class ONNX:
    def __init__(
        self,
        wrapper: Wrapper,
        onnx_path: str = None,
    ):
        self.wrapper = wrapper
        self.model = wrapper.model.cpu()
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
        self.wrapper.pipeline.onnx = self
        del self.wrapper.model, self.wrapper.pipeline.model
        return self

    def inference(self, new_inputs: Dict[str, np.ndarray]) -> tensor_dict_type:
        assert self.ort_session is not None
        ort_inputs = {
            node.name: to_standard(new_inputs[node.name])
            for node in self.ort_session.get_inputs()
        }
        outputs = dict(zip(self.output_names, self.ort_session.run(None, ort_inputs)))
        return {k: to_torch(v) for k, v in outputs.items()}


def make_toy_model(
    model: str = "fcnn",
    config: Dict[str, Any] = None,
    *,
    task_type: str = "reg",
    data_tuple: Tuple[data_type, data_type] = None,
) -> Wrapper:
    if config is None:
        config = {}
    config.setdefault("min_epoch", 1)
    config.setdefault("num_epoch", 2)
    config.setdefault("max_epoch", 4)
    if data_tuple is not None:
        x, y = data_tuple
        assert isinstance(x, list)
    else:
        if task_type == "reg":
            x, y = [[0]], [[1]]
        else:
            x, y = [[0], [1]], [[1], [0]]
    data_tuple = x, y
    base_config = {
        "model": model,
        "model_config": {
            "hidden_units": [100],
            "mapping_configs": {"dropout": 0.0, "batch_norm": False},
        },
        "cv_split": 0.0,
        "trigger_logging": False,
        "min_epoch": 250,
        "num_epoch": 500,
        "max_epoch": 1000,
        "optimizer": "sgd",
        "optimizer_config": {"lr": 0.01},
        "task_type": task_type,
        "data_config": {
            "valid_columns": list(range(len(x[0]))),
            "label_process_method": "identical",
        },
        "verbose_level": 0,
    }
    updated = update_dict(config, base_config)
    return make(**updated).fit(*data_tuple)


__all__ = [
    "ONNX",
    "make_toy_model",
]
