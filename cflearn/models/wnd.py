from .base import ModelBase


@ModelBase.register("wnd")
@ModelBase.register_pipe("fcnn", transform="embedding")
@ModelBase.register_pipe("linear", transform="one_hot_only")
class WideAndDeep(ModelBase):
    pass


__all__ = ["WideAndDeep"]
