import torch
import torch.nn as nn

from functools import partial
from torch.nn.functional import softplus


class BN(nn.BatchNorm1d):
    def forward(self, net):
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        net = super().forward(net)
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        return net
