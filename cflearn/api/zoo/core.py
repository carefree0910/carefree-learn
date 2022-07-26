import os
import json
import torch

from abc import ABC
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import NamedTuple
from cftool.misc import update_dict
from cftool.types import tensor_dict_type

from ...pipeline import DLPipeline
from ...pipeline import PipelineProtocol
from ...protocol import ModelProtocol
from ...constants import WARNING_PREFIX
from ...constants import DEFAULT_ZOO_TAG
from ...misc.toolkit import inject_debug
from ...misc.toolkit import download_model
from ...misc.toolkit import download_data_info


root = os.path.dirname(__file__)
configs_root = os.path.join(root, "configs")


class ParsedModel(NamedTuple):
    json_path: str
    download_name: str


def _parse_model(model: str) -> ParsedModel:
    tag = DEFAULT_ZOO_TAG
    model_type, model_name = model.split("/")
    download_name = model_name
    if "." in model_name:
        model_name, tag = model_name.split(".")
    json_folder = os.path.join(configs_root, model_type, model_name)
    json_path = os.path.join(json_folder, f"{tag}.json")
    if not os.path.isfile(json_path):
        json_path = os.path.join(json_folder, f"{DEFAULT_ZOO_TAG}.json")
    return ParsedModel(json_path, download_name)


def _parse_config(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r") as f:
        config = json.load(f)
    bases = config.pop("__bases__", [])
    for base in bases[::-1]:
        parsed = _parse_model(base)
        config = update_dict(config, _parse_config(parsed.json_path))
    return config


class ZooBase(ABC):
    def __init__(
        self,
        model: Optional[str] = None,
        *,
        data_info: Optional[Dict[str, Any]] = None,
        json_path: Optional[str] = None,
        no_build: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ):
        self.download_name = None
        # load json
        if json_path is None:
            if model is None:
                raise ValueError("either `model` or `json_path` should be provided")
            parsed = _parse_model(model)
            json_path = parsed.json_path
            self.download_name = parsed.download_name
        self.json_path = json_path
        self.config = _parse_config(json_path)
        if self.download_name is None:
            self.download_name = self.config.pop("tag", None)
        self.err_msg_fmt = f"`{'{}'}` should be provided in '{json_path}'"
        # get pipeline
        self.pipeline_name = self.config.pop("pipeline", None)
        if self.pipeline_name is None:
            raise ValueError(self.err_msg_fmt.format("pipeline"))
        # handle debug
        if debug:
            inject_debug(kwargs)

        # handle requires
        def _inject_requires(d: Dict[str, Any], local_requires: Dict[str, Any]) -> None:
            for k, v in local_requires.items():
                kd = d.setdefault(k, {})
                if isinstance(v, dict):
                    _inject_requires(kd, v)
                    continue
                assert isinstance(v, list), "requirements should be a list"
                for vv in v:
                    if vv in kd:
                        continue
                    required = kwargs.pop(vv, None)
                    if required is None:
                        raise ValueError(f"'{vv}' should be provided in `kwargs`")
                    kd[vv] = required

        requires = self.config.pop("__requires__", {})
        _inject_requires(kwargs, requires)
        # build
        update_dict(kwargs, self.config)
        if no_build:
            self.m = None
        else:
            self.m = DLPipeline.make(self.pipeline_name, self.config)
            if data_info is None:
                if self.download_name is None:
                    data_info = {}
                else:
                    try:
                        with open(download_data_info(self.download_name), "r") as f:
                            data_info = json.load(f)
                    except ValueError:
                        data_info = {}
            try:
                self.m.build(data_info)
            except Exception as err:
                raise ValueError(f"Failed to build '{model}': {err}")

    @classmethod
    def load_pipeline(
        cls,
        model: Optional[str] = None,
        *,
        data_info: Optional[Dict[str, Any]] = None,
        json_path: Optional[str] = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> PipelineProtocol:
        zoo = cls(
            model,
            data_info=data_info,
            json_path=json_path,
            debug=debug,
            **kwargs,
        )
        assert zoo.m is not None
        return zoo.m


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
        pretrained: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> DLPipeline:
        zoo = cls(
            model,
            data_info=data_info,
            json_path=json_path,
            debug=debug,
            **kwargs,
        )
        if pretrained:
            zoo.load_pretrained()
        return zoo.m

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
        if pretrained:
            zoo.load_pretrained()
        return zoo.m.model

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
        input_sample: Optional[tensor_dict_type] = None,
        num_samples: Optional[int] = None,
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
            input_sample=input_sample,
            num_samples=num_samples,
            verbose=verbose,
        )
        return zoo.m


__all__ = ["DLZoo"]
