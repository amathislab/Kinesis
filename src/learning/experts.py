# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.

import torch.nn as nn
from src.learning.mlp import MLP
from src.learning.running_norm import RunningNorm
import torch

class Experts(nn.Module):
    def __init__(self, cfg, action_dim, state_dim, num_experts, freeze=True):
        super().__init__()
        self.norm = RunningNorm(state_dim)

        mlp_hsize = cfg.learning.mlp.units
        mlp_htype = cfg.learning.mlp.activation

        self.experts = nn.ModuleList()
        for i in range(num_experts):
            norm = RunningNorm(state_dim)
            action_mean = nn.Linear(mlp_hsize[-1], action_dim)
            action_mean.weight.data.mul_(0.1)
            action_mean.bias.data.mul_(0.0)
            net = MLP(state_dim, mlp_hsize, mlp_htype)
            # The first time we construct the MoE network, we need to load the expert weights from their respective checkpoints
            if cfg.epoch == 0:
                state = torch.load(cfg.run.expert_path + f"expert_{i}" + "/model.pth")
                net_state = state["policy"]
                # Keep only the elements starting with "net."
                net_state = {k[4:]: v for k, v in net_state.items() if k.startswith("net.")}
                net.load_state_dict(net_state)
                action_mean.weight.data = state["policy"]["action_mean.weight"]
                action_mean.bias.data = state["policy"]["action_mean.bias"]
                norm.n = state["policy"]["norm.n"]
                norm.mean = state["policy"]["norm.mean"]
                norm.var = state["policy"]["norm.var"]
                norm.std = state["policy"]["norm.std"]
                print(f"Expert {i} -- epoch {state['epoch']} loaded.")

                del state
                del net_state
            
            self.experts.append(
                nn.Sequential(
                    norm,
                    net,
                    action_mean
                )
            )

        # Freeze experts
        if freeze:
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = False
