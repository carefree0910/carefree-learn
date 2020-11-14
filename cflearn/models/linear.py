from .base import ModelBase


@ModelBase.register("linear")
@ModelBase.register_pipe("linear")
class LinearModel(ModelBase):
    pass


__all__ = ["LinearModel"]
