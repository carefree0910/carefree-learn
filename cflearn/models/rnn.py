from .base import ModelBase


@ModelBase.register("rnn")
@ModelBase.register_pipe("rnn", head="fcnn")
class RNN(ModelBase):
    pass


__all__ = ["RNN"]
