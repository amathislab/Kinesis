# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.

import torch.nn as nn
from src.learning.policy import Policy
from src.learning.mlp import MLP
from src.learning.running_norm import RunningNorm
import torch

from torch.distributions import MultivariateNormal

class PolicyLattice(Policy):
    def __init__(self, cfg, action_dim, latent_dim, state_dim, net_out_dim=None):
        super().__init__()
        self.type = "lattice"
        self.norm = RunningNorm(state_dim)
        
        policy_hsize = cfg.learning.mlp.units
        policy_htype = cfg.learning.mlp.activation
        fix_std = cfg.learning.fix_std
        log_std = cfg.learning.log_std
        self.net = net = MLP(state_dim, policy_hsize, policy_htype)
        
        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.action_mean = nn.Linear(net_out_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.log_std = nn.Parameter(
            torch.ones(1, action_dim + latent_dim) * log_std, requires_grad=not fix_std
        )

        self.action_dim = action_dim
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.norm(x)
        x = self.net(x)
        action_mean = self.action_mean(x)
        std = torch.exp(self.log_std)
        action_var = std[:, :self.action_dim] ** 2
        latent_var = std[:, self.action_dim:] ** 2
        sigma_mat = (self.action_mean.weight * latent_var[..., None, :]).matmul(self.action_mean.weight.T)
        sigma_mat[..., torch.arange(self.action_dim), torch.arange(self.action_dim)] += action_var
        self.lattice_dist = MultivariateNormal(action_mean, sigma_mat)
        return self.lattice_dist
    
    def select_action(self, x, mean_action=False):
        dist = self.forward(x) 
        action = dist.loc if mean_action else dist.rsample()
        return action
    
    def get_log_prob(self, x, value):
        dist = self.forward(x)
        return dist.log_prob(value).unsqueeze(1)

    def get_fim(self, x):
        dist = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), dist.loc, {"std_id": std_id, "std_index": std_index}
