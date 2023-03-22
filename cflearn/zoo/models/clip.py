from typing import Any
from typing import Dict
from typing import Callable
from cftool.types import tensor_dict_type

from .schema import IDLZooModelLoader


def get_callback(use_vision: bool, use_text: bool) -> Callable:
    def callback(states: tensor_dict_type) -> tensor_dict_type:
        if not use_vision:
            for key in list(states.keys()):
                if key.startswith("vit"):
                    states.pop(key)
        if not use_text:
            for key in list(states.keys()):
                if (
                    key.startswith("token_embedding")
                    or key.startswith("text_transformer")
                    or key.startswith("text_projection")
                ):
                    states.pop(key)
        return states

    return callback


@IDLZooModelLoader.register("multimodal/clip")
@IDLZooModelLoader.register("multimodal/clip.large")
@IDLZooModelLoader.register("multimodal/clip.chinese")
@IDLZooModelLoader.register("multimodal/clip.open_clip_ViT_H_14")
class CLIPModelLoader(IDLZooModelLoader):
    def permute_kwargs(self, kwargs: Dict[str, Any]) -> None:
        use_vision = kwargs.pop("use_vision", True)
        use_text = kwargs.pop("use_text", True)
        if not use_vision or not use_text:
            kwargs["states_callback"] = get_callback(use_vision, use_text)
            model_config = kwargs.setdefault("model_config", {})
            model_config.setdefault("use_vision", use_vision)
            model_config.setdefault("use_text", use_text)


__all__ = [
    "CLIPModelLoader",
]
