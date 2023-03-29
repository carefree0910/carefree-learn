import os
import json
import torch

from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Optional
from typing import NamedTuple
from cftool.misc import update_dict
from cftool.misc import safe_execute
from cftool.misc import shallow_copy_dict
from cftool.types import tensor_dict_type

from .models import dl_zoo_model_loaders
from ..schema import IData
from ..schema import DLConfig
from ..pipeline import Pipeline
from ..pipeline import PipelineTypes
from ..pipeline import TrainingPipeline
from ..pipeline import DLInferencePipeline
from ..constants import DEFAULT_ZOO_TAG
from ..parameters import OPT
from ..misc.toolkit import download_model


root = os.path.dirname(__file__)
configs_root = os.path.join(root, "configs")
TPipeline = Union[TrainingPipeline, DLInferencePipeline]


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


class DLZoo:
    model_dir = "models_v0.4.x"

    # api

    @classmethod
    def load_config(
        cls,
        model: Optional[str] = None,
        *,
        json_path: Optional[str] = None,
        download_name: Optional[str] = None,
        **kwargs: Any,
    ) -> DLConfig:
        return cls._load_config(
            model,
            json_path=json_path,
            download_name=download_name,
            **kwargs,
        )[0]

    @classmethod
    def load_pipeline(
        cls,
        model: Optional[str] = None,
        *,
        data: Optional[IData] = None,
        states: Optional[tensor_dict_type] = None,
        pretrained: bool = False,
        pipeline_type: PipelineTypes = PipelineTypes.DL_INFERENCE,
        download_name: Optional[str] = None,
        model_dir: Optional[str] = None,
        json_path: Optional[str] = None,
        **kwargs: Any,
    ) -> TPipeline:
        states_callback = kwargs.pop("states_callback", None)
        # get config
        config, download_name = cls._load_config(
            model,
            json_path=json_path,
            download_name=download_name,
            **kwargs,
        )
        # handle states
        if states is None and pretrained:
            if download_name is None:
                err_msg = f"`{'tag'}` should be provided in '{json_path}' when `pretrained` is True"
                raise ValueError(err_msg)
            if model_dir is None:
                model_dir = cls.model_dir
            root = os.path.join(OPT.cache_dir, model_dir)
            states_path = download_model(download_name, root=root)
            states = torch.load(states_path, map_location="cpu")
        if states is not None and states_callback is not None:
            states = states_callback(states)
        # build
        m_base: Type[Pipeline] = Pipeline.get(pipeline_type)
        if issubclass(m_base, DLInferencePipeline):
            return m_base.build_with(config, states)
        m = m_base.init(config)
        if isinstance(m, TrainingPipeline):
            if states is not None:
                if data is None:
                    msg = "`data` needs to be provided when loading `TrainingPipeline` with `states`"
                    raise ValueError(msg)
                m.prepare(data)
                m.build_model.model.load_state_dict(states)
        return m

    # internal

    @classmethod
    def _load_config(
        cls,
        model: Optional[str] = None,
        *,
        json_path: Optional[str] = None,
        download_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[DLConfig, Optional[str]]:
        # loader
        if model is not None:
            loader_base = dl_zoo_model_loaders.get(model)
            if loader_base is not None:
                loader_base().permute_kwargs(kwargs)
        # load json
        if json_path is None:
            if model is None:
                raise ValueError("either `model` or `json_path` should be provided")
            parsed = _parse_model(model)
            json_path = parsed.json_path
            if download_name is None:
                download_name = parsed.download_name
        parsed_config = _parse_config(json_path)
        if download_name is None:
            download_name = parsed_config.pop("tag", None)
        # handle requires

        def _example(l: str, r: str, h: List[str]) -> str:
            key = h.pop(0)
            if not h:
                return f"{l}{key}=...{r}"
            return _example(f"{l}{key}=dict(", f"){r}", h)

        def _inject_requires(
            inc: Dict[str, Any],
            reference: Dict[str, Any],
            local_requires: Dict[str, Any],
            hierarchy: Optional[str],
        ) -> None:
            for k, v in local_requires.items():
                k_hierarchy = k if hierarchy is None else f"{hierarchy} -> {k}"
                ki = inc.setdefault(k, {})
                kr = reference.setdefault(k, {})
                if isinstance(v, dict):
                    _inject_requires(ki, kr, v, k_hierarchy)
                    continue
                assert isinstance(v, list), "requirements should be a list"
                for vv in v:
                    shortcut = kwargs.pop(vv, no_shortcut_token)
                    if shortcut != no_shortcut_token:
                        ki[vv] = shortcut
                        continue
                    if vv not in kr:
                        example = _example("", "", k_hierarchy.split(" -> ") + [vv])
                        raise ValueError(
                            f"Failed to build '{model}': "
                            f"'{vv}' should be provided in `{k_hierarchy}`, for example:\n"
                            f'* cflearn.api.from_zoo("{model}", {example})\n'
                            f'* cflearn.ZooBase.load_pipeline("{model}", {example})\n'
                            f"\nOr you can use shortcut for simplicity:\n"
                            f'* cflearn.api.from_zoo("{model}", {vv}=...)\n'
                            f'* cflearn.ZooBase.load_pipeline("{model}", {vv}=...)\n'
                        )

        no_shortcut_token = "@#$no_shortcut$#@"
        requires = parsed_config.pop("__requires__", {})
        merged_config = shallow_copy_dict(parsed_config)
        update_dict(shallow_copy_dict(kwargs), merged_config)
        increment: Dict[str, Any] = {}
        _inject_requires(increment, merged_config, requires, None)
        raw_config = shallow_copy_dict(parsed_config)
        update_dict(kwargs, raw_config)
        update_dict(increment, raw_config)
        config_type = raw_config.pop("config_type", "dl")
        config = safe_execute(DLConfig.get(config_type), raw_config)
        return config, download_name


__all__ = [
    "DLZoo",
]
