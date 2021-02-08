from .base import ModelBase


@ModelBase.register("transformer")
@ModelBase.register_pipe("transformer", head="linear")
class Transformer(ModelBase):
    pass


__all__ = ["Transformer"]
