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


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'gelu':
            self.activation = torch.nn.GELU()
        elif activation == 'silu':
            self.activation = torch.nn.SiLU()

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x
