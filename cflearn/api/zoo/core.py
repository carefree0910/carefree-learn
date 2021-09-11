import os
import json

from abc import abstractmethod
from abc import ABC
from typing import Any
from typing import Optional
from cftool.misc import update_dict

from ..basic import make
from ..internal_.pipeline import ModelProtocol
from ..internal_.pipeline import PipelineProtocol
from ..cv.pipeline import SimplePipeline as CVPipeline
from ...misc.toolkit import download_model


root = os.path.dirname(__file__)
configs_root = os.path.join(root, "configs")


class ZooBase(ABC):
    def __init__(
        self,
        model: Optional[str] = None,
        *,
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

    @abstractmethod
    def get_model(self, *, pretrained: bool = False) -> ModelProtocol:
        pass

    @classmethod
    def load_model(
        cls,
        model: Optional[str] = None,
        *,
        json_path: Optional[str] = None,
        pretrained: bool = False,
        **kwargs: Any,
    ) -> ModelProtocol:
        zoo = cls(model, json_path=json_path, **kwargs)
        return zoo.get_model(pretrained=pretrained)

    @classmethod
    def load_pipeline(
        cls,
        model: Optional[str] = None,
        *,
        json_path: Optional[str] = None,
        **kwargs: Any,
    ) -> PipelineProtocol:
        return cls(model, json_path=json_path, **kwargs).m


class CVZoo(ZooBase):
    m: CVPipeline

    def get_model(self, *, pretrained: bool = False) -> ModelProtocol:
        m = ModelProtocol.make(self.m.model_name, config=self.m.model_config)
        if pretrained:
            if self.tag is None:
                err_msg = self.err_msg_fmt.format("tag")
                raise ValueError(f"{err_msg} when `pretrained` is True")
            m.load_state_dict(download_model(self.tag))
        return m


__all__ = [
    "CVZoo",
]
