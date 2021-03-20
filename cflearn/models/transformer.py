from .base import ModelBase


@ModelBase.register("transformer")
@ModelBase.register_pipe("transformer", head="fcnn", head_config="highway")
class Transformer(ModelBase):
    pass


__all__ = ["Transformer"]
