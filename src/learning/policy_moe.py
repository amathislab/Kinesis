# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.

import torch.nn as nn
from src.learning.policy import Policy
from src.learning.mlp import MLP
from src.learning.running_norm import RunningNorm
from src.learning.experts import Experts
import torch
import numpy as np

class PolicyMOE(Policy):
    # A mixture of experts policy
    def __init__(self, cfg, action_dim, state_dim, net_out_dim=None, freeze=True):
        super().__init__()
        self.type = "moe"
        self.norm = RunningNorm(state_dim)
        if freeze and cfg.epoch == 0:
            state = torch.load(cfg.run.expert_path + "expert_0" + "/model.pth")
            self.norm.n = state["policy"]["norm.n"]
            self.norm.mean = state["policy"]["norm.mean"]
            self.norm.var = state["policy"]["norm.var"]
            self.norm.std = state["policy"]["norm.std"]

            del state

        policy_hsize = cfg.learning.moe.units
        policy_htype = cfg.learning.moe.activation

        self.gate = nn.Sequential(
            MLP(state_dim, policy_hsize, policy_htype),
            nn.Linear(policy_hsize[-1], cfg.num_experts),
            nn.Softmax(dim=1)
        )

        self.experts = Experts(cfg, action_dim, state_dim, cfg.num_experts, freeze)

    def forward(self, x):
        gating_input = self.norm(x)
        weight = self.gate(gating_input)
        # The weight is the probability of choosing each expert
        # One-hot distribution
        action_dist = torch.distributions.Categorical(weight)
        return action_dist
    
    def select_action(self, x, mean_action=False):
        dist = self.forward(x)
        expert_idx = dist.sample()
        expert_actions = [expert(x) for expert in self.experts.experts]
        action = expert_actions[expert_idx]
        return action
    
    def get_log_prob(self, x, value):
        dist = self.forward(x)
        expert_actions = [expert(x) for expert in self.experts.experts]
        expert_ids = []
        
        for i in range(value.shape[0]):
            distances = []
            for expert_action in expert_actions:
                distances.append(torch.sum(torch.abs(value[i] - expert_action[i])))
            expert_ids.append(torch.argmin(torch.tensor(distances)))
        
        expert_ids = torch.tensor(expert_ids).to(x.device)
        return dist.log_prob(expert_ids).unsqueeze(1)

        