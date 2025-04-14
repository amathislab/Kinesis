# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.
#
# 2. PyTorch-RL (https://github.com/Khrylx/PyTorch-RL)
#   Copyright (c) 2020 Ye Yuan

import torch.nn as nn
import torch


class Value(nn.Module):
    def __init__(self, net, net_out_dim=None):
        super().__init__()
        self.net = net
        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.value_head = nn.Linear(net_out_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.net(x)
        value = self.value_head(x)
        return value
