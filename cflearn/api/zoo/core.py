import os
import dill
import json
import torch

from abc import ABC
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from cftool.misc import update_dict

from ..basic import make
from ...types import tensor_dict_type
from ...constants import WARNING_PREFIX
from ...constants import DEFAULT_ZOO_TAG
from ..internal_.pipeline import DLPipeline
from ..internal_.pipeline import ModelProtocol
from ..internal_.pipeline import PipelineProtocol
from ...misc.toolkit import download_model
from ...misc.toolkit import download_data_info
from ...misc.toolkit import download_tokenizer


root = os.path.dirname(__file__)
configs_root = os.path.join(root, "configs")


# tokenizers


def load_tokenizer(name: str) -> Any:
    with open(download_tokenizer(name), "rb") as f:
        return dill.load(f)


# models


class ZooBase(ABC):
    def __init__(
        self,
        model: Optional[str] = None,
        *,
        data_info: Optional[Dict[str, Any]] = None,
        json_path: Optional[str] = None,
        **kwargs: Any,
    ):
        tag = DEFAULT_ZOO_TAG
        self.download_name = None
        if json_path is None:
            if model is None:
                raise ValueError("either `model` or `json_path` should be provided")
            model_type, model_name = model.split("/")
            self.download_name = model_name
            if "." in model_name:
                model_name, tag = model_name.split(".")
            json_folder = os.path.join(configs_root, model_type, model_name)
            json_path = os.path.join(json_folder, f"{tag}.json")
            if not os.path.isfile(json_path):
                json_path = os.path.join(json_folder, f"{DEFAULT_ZOO_TAG}.json")
        self.json_path = json_path
        with open(json_path, "r") as f:
            self.config = json.load(f)
        if self.download_name is None:
            self.download_name = self.config.pop("tag", None)
        self.err_msg_fmt = f"`{'{}'}` should be provided in '{json_path}'"
        self.pipeline_name = self.config.pop("pipeline", None)
        if self.pipeline_name is None:
            raise ValueError(self.err_msg_fmt.format("pipeline"))
        update_dict(kwargs, self.config)
        self.m = make(self.pipeline_name, config=self.config)
        if data_info is None:
            if self.download_name is None:
                data_info = {}
            else:
                try:
                    with open(download_data_info(self.download_name), "r") as f:
                        data_info = json.load(f)
                except ValueError:
                    print(
                        f"{WARNING_PREFIX}`data_info` of '{self.download_name}' "
                        "does not exist on the remote server, "
                        "empty `data_info` will be used"
                    )
                    data_info = {}
        self.m.build(data_info)

    @classmethod
    def load_pipeline(
        cls,
        model: Optional[str] = None,
        *,
        data_info: Optional[Dict[str, Any]] = None,
        json_path: Optional[str] = None,
        **kwargs: Any,
    ) -> PipelineProtocol:
        return cls(model, data_info=data_info, json_path=json_path, **kwargs).m


class DLZoo(ZooBase):
    m: DLPipeline

    def load_pretrained(self) -> ModelProtocol:
        if self.download_name is None:
            err_msg = self.err_msg_fmt.format("tag")
            raise ValueError(f"{err_msg} when `pretrained` is True")
        m = self.m.model
        m.load_state_dict(torch.load(download_model(self.download_name)))
        return m

    @classmethod
    def load_pipeline(
        cls,
        model: Optional[str] = None,
        *,
        data_info: Optional[Dict[str, Any]] = None,
        json_path: Optional[str] = None,
        **kwargs: Any,
    ) -> DLPipeline:
        return super().load_pipeline(  # type: ignore
            model,
            data_info=data_info,
            json_path=json_path,
            **kwargs,
        )

    @classmethod
    def load_model(
        cls,
        model: Optional[str] = None,
        *,
        data_info: Optional[Dict[str, Any]] = None,
        json_path: Optional[str] = None,
        pretrained: bool = False,
        **kwargs: Any,
    ) -> ModelProtocol:
        kwargs.setdefault("in_loading", True)
        zoo = cls(model, data_info=data_info, json_path=json_path, **kwargs)
        if not pretrained:
            return zoo.m.model
        return zoo.load_pretrained()

    @classmethod
    def dump_onnx(
        cls,
        model: str,
        export_folder: str,
        dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
        *,
        data_info: Optional[Dict[str, Any]] = None,
        json_path: Optional[str] = None,
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
    ) -> DLPipeline:
        kwargs["in_loading"] = True
        zoo = cls(model, data_info=data_info, json_path=json_path, **kwargs)
        try:
            zoo.load_pretrained()
        except ValueError:
            print(
                f"{WARNING_PREFIX}no pretrained models are available for '{model}', "
                f"so onnx will not be dumped"
            )
            return zoo.m
        zoo.m.to_onnx(
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
        )
        return zoo.m


__all__ = [
    "load_tokenizer",
    "DLZoo",
]
