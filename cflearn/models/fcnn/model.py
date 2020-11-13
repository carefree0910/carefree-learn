from ..base import ModelBase


@ModelBase.register("fcnn")
@ModelBase.register_pipe("fcnn")
class FCNN(ModelBase):
    pass


__all__ = ["FCNN"]
