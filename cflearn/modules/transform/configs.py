from typing import Any
from typing import Dict

from ...configs import Configs


transform_config_mapping = {
    "default": {"one_hot": True, "embedding": True, "only_categorical": False},
    "categorical_only": {"one_hot": True, "embedding": True, "only_categorical": True},
    "one_hot": {"one_hot": True, "embedding": False, "only_categorical": False},
    "one_hot_only": {"one_hot": True, "embedding": False, "only_categorical": True},
    "embedding": {"one_hot": False, "embedding": True, "only_categorical": False},
    "embedding_only": {"one_hot": False, "embedding": True, "only_categorical": True},
    "numerical": {"one_hot": False, "embedding": False, "only_categorical": False},
}


@Configs.register("transform", "default")
class DefaultTransformConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return transform_config_mapping["default"]


@Configs.register("transform", "categorical_only")
class CategoricalOnlyTransformConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return transform_config_mapping["categorical_only"]


@Configs.register("transform", "one_hot")
class OneHotTransformConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return transform_config_mapping["one_hot"]


@Configs.register("transform", "one_hot_only")
class OneHotOnlyTransformConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return transform_config_mapping["one_hot_only"]


@Configs.register("transform", "embedding")
class EmbeddingTransformConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return transform_config_mapping["embedding"]


@Configs.register("transform", "embedding_only")
class EmbeddingOnlyTransformConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return transform_config_mapping["embedding_only"]


@Configs.register("transform", "numerical")
class NumericalTransformConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return transform_config_mapping["numerical"]


__all__ = [
    "transform_config_mapping",
    "DefaultTransformConfig",
    "CategoricalOnlyTransformConfig",
    "OneHotTransformConfig",
    "OneHotOnlyTransformConfig",
    "EmbeddingTransformConfig",
    "EmbeddingOnlyTransformConfig",
    "NumericalTransformConfig",
]
