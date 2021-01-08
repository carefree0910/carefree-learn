from .base import ModelBase


@ModelBase.register("rnn")
@ModelBase.register_pipe("rnn", head="fcnn")
class RNN(ModelBase):
    pass


@ModelBase.register("tree_rnn")
@ModelBase.register_pipe("rnn_dndf", extractor="rnn", head="dndf")
@ModelBase.register_pipe("rnn_fcnn", extractor="rnn", head="fcnn", head_config="pruned")
class TreeRNN(ModelBase):
    pass


__all__ = ["RNN"]
