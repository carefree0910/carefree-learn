from .base import ModelBase
from .base import SiameseModelBase


@ModelBase.register("rnn")
@ModelBase.register_pipe("rnn", head="fcnn")
class RNN(ModelBase):
    pass


@SiameseModelBase.register("siamese_rnn")
@SiameseModelBase.register_pipe("rnn", head="fcnn")
class SiameseRNN(SiameseModelBase):
    pass


@ModelBase.register("tree_rnn")
@ModelBase.register_pipe("rnn_dndf", extractor="rnn", head="dndf")
@ModelBase.register_pipe("rnn_fcnn", extractor="rnn", head="fcnn", head_config="pruned")
class TreeRNN(ModelBase):
    pass


__all__ = [
    "RNN",
    "TreeRNN",
    "SiameseRNN",
]
