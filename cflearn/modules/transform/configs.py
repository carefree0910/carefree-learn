from typing import Any
from typing import Dict

from ...misc.configs import Configs


@Configs.register("transform", "default")
class DefaultTransformConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return {"one_hot": True, "embedding": True, "only_categorical": False}


@Configs.register("transform", "categorical_only")
class CategoricalOnlyTransformConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return {"one_hot": True, "embedding": True, "only_categorical": True}


@Configs.register("transform", "one_hot")
class OneHotTransformConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return {"one_hot": True, "embedding": False, "only_categorical": False}


@Configs.register("transform", "one_hot_only")
class OneHotOnlyTransformConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return {"one_hot": True, "embedding": False, "only_categorical": True}


@Configs.register("transform", "embedding")
class EmbeddingTransformConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return {"one_hot": False, "embedding": True, "only_categorical": False}


@Configs.register("transform", "embedding_only")
class EmbeddingOnlyTransformConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return {"one_hot": False, "embedding": True, "only_categorical": True}


@Configs.register("transform", "numerical")
class NumericalTransformConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return {"one_hot": False, "embedding": False, "only_categorical": False}


__all__ = [
    "DefaultTransformConfig",
    "CategoricalOnlyTransformConfig",
    "OneHotTransformConfig",
    "OneHotOnlyTransformConfig",
    "EmbeddingTransformConfig",
    "EmbeddingOnlyTransformConfig",
    "NumericalTransformConfig",
]
