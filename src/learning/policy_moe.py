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

        for i, expert in enumerate(self.experts.experts):
            if i == expert_idx.item():
                action = expert(x)
                break

        return action, expert_idx
    
    def get_log_prob(self, x, expert_idx):
        dist = self.forward(x)
        return dist.log_prob(expert_idx).unsqueeze(1)

# Define a MOE policy that also receives the previous selected expert index in its input
class PolicyMOEWithPrev(Policy):
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
            MLP(state_dim + cfg.num_experts, policy_hsize, policy_htype),
            nn.Linear(policy_hsize[-1], cfg.num_experts),
            nn.Softmax(dim=1)
        )

        self.experts = Experts(cfg, action_dim, state_dim, cfg.num_experts, freeze)

        print(f"PolicyMOEWithPrev initialized with {cfg.num_experts} experts and input dimension {state_dim + cfg.num_experts}")

    def forward(self, x, prev_expert_idx):
        # Put the input in the same device as the gate
        device = self.gate[1].bias.device
        x = x.to(device)
        prev_expert_idx = prev_expert_idx.to(device)
        gating_input = self.norm(x)
        # Concatenate the previous expert index to the input
        gating_input = torch.cat((gating_input, prev_expert_idx.reshape(gating_input.shape[0], -1)), dim=1)
        weight = self.gate(gating_input)
        # The weight is the probability of choosing each expert
        # One-hot distribution
        action_dist = torch.distributions.Categorical(weight)
        return action_dist
    
    def select_action(self, x, prev_expert_idx_oh, mean_action=False):
        dist = self.forward(x, prev_expert_idx_oh)
        expert_idx = dist.sample()

        for i, expert in enumerate(self.experts.experts):
            if i == expert_idx.item():
                action = expert(x)
                break

        return action, expert_idx

    def get_log_prob(self, x, prev_expert_idx, expert_idx):
        dist = self.forward(x, prev_expert_idx)
        return dist.log_prob(expert_idx).unsqueeze(1)