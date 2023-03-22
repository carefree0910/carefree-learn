import torch

from torch import nn
from cftool.misc import print_info


def load_weights(model: nn.Module, ckpt_path: str, verbose: bool = False) -> None:
    if verbose:
        print_info(f"Load checkpoint from path: {ckpt_path}")
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(ckpt_path, map_location="cpu")
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)
