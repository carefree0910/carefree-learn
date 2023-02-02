import torch.nn as nn

from ..mconfigs import ALL_MCONFIGS
from ..utils.misc import load_weights


def load_model(model_type: str, ckpt_path: str, verbose: bool = False) -> nn.Module:
    net = ALL_MCONFIGS[model_type]["model"](**ALL_MCONFIGS[model_type]["params"])
    load_weights(net, ckpt_path, verbose=verbose)
    return net
