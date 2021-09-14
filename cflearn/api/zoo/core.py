import os
import dill
import json
import torch

from abc import ABC
from typing import Any
from typing import Dict
from typing import Optional
from cftool.misc import update_dict

from ..basic import make
from ...constants import WARNING_PREFIX
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
        if json_path is None:
            if model is None:
                raise ValueError("either `model` or `json_path` should be provided")
            if "/" not in model:
                model = f"{model}/baseline"
            model_type, model_name = model.split("/")
            json_path = os.path.join(configs_root, model_type, f"{model_name}.json")
        self.json_path = json_path
        with open(json_path, "r") as f:
            self.config = json.load(f)
        self.err_msg_fmt = f"`{'{}'}` should be provided in '{json_path}'"
        self.tag = self.config.pop("tag", None)
        self.pipeline_name = self.config.pop("pipeline", None)
        if self.pipeline_name is None:
            raise ValueError(self.err_msg_fmt.format("pipeline"))
        update_dict(kwargs, self.config)
        self.m = make(self.pipeline_name, config=self.config)
        if data_info is None:
            try:
                with open(download_data_info(self.tag), "r") as f:
                    data_info = json.load(f)
            except ValueError:
                print(
                    f"{WARNING_PREFIX}`data_info` of '{self.tag}' does not exist "
                    f"on the remote server, empty `data_info` will be used"
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
        m = zoo.m.model
        if pretrained:
            if zoo.tag is None:
                err_msg = zoo.err_msg_fmt.format("tag")
                raise ValueError(f"{err_msg} when `pretrained` is True")
            m.load_state_dict(torch.load(download_model(zoo.tag)))
        return m


__all__ = [
    "load_tokenizer",
    "DLZoo",
]