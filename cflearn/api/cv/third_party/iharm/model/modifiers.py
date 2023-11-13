import torch


class LRMult:
    def __init__(self, lr_mult: float = 1.0):
        self.lr_mult = lr_mult

    def __call__(self, m: torch.nn.Module) -> None:
        if getattr(m, "weight", None) is not None:
            m.weight.lr_mult = self.lr_mult
        if getattr(m, "bias", None) is not None:
            m.bias.lr_mult = self.lr_mult
