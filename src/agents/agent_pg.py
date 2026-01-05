# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.

from typing import Optional
import torch
import logging

from src.learning.policy_moe import PolicyMOEWithPrev
from src.learning.learning_utils import to_train, to_test
from src.learning.learning_utils import estimate_advantages
from src.agents.agent import Agent

import time

logger = logging.getLogger(__name__)


class AgentPG(Agent):
    """
    Implements the policy gradient algorithm within the A2C framework.
    """

    def __init__(
        self,
        tau: float = 0.95,
        optimizer_policy: Optional[torch.optim.Optimizer] = None,
        optimizer_value: Optional[torch.optim.Optimizer] = None,
        opt_num_epochs: int = 1,
        value_opt_niter: int = 1,
        **kwargs
    ):
        """
        Initialize the Policy Gradient Agent.

        Args:
            tau (float): GAE parameter for bias-variance trade-off.
            optimizer_policy (torch.optim.Optimizer, optional): Optimizer for the policy network.
            optimizer_value (torch.optim.Optimizer, optional): Optimizer for the value network.
            opt_num_epochs (int): Number of epochs for policy updates.
            value_opt_niter (int): Number of iterations for value network updates.
            **kwargs: Additional parameters for the base Agent class.
        """
        super().__init__(**kwargs)
        self.tau = tau
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.opt_num_epochs = opt_num_epochs
        self.value_opt_niter = value_opt_niter

    def update_value(self, states: torch.Tensor, returns: torch.Tensor) -> float:
        """
        Update the critic (value network) by minimizing the MSE between predicted values and returns.

        Args:
            states (torch.Tensor): Tensor of states.
            returns (torch.Tensor): Tensor of target returns.

        Returns:
            float: The average value loss.
        """
        value_loss_val = 0.0
        for _ in range(self.value_opt_niter):
            values_pred = self.value_net(states)
            value_loss = (values_pred - returns).pow(2).mean()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()
            value_loss_val += value_loss.item()
        return value_loss_val / self.value_opt_niter

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        exps: torch.Tensor,
        expert_indices: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Update the policy network using advantage-weighted log probabilities.

        Args:
            states (torch.Tensor): Tensor of states.
            actions (torch.Tensor): Tensor of actions taken.
            returns (torch.Tensor): Tensor of target returns.
            advantages (torch.Tensor): Tensor of advantage estimates.
            exps (torch.Tensor): Tensor indicating exploration flags.
            expert_indices (Optional[torch.Tensor]): Tensor of expert indices for actions, if applicable.

        Returns:
            dict: Dictionary containing training metrics.
        """
        # use a2c by default
        ind = exps.nonzero().squeeze(1)
        policy_loss_val = 0.0
        value_loss_val = 0.0

        for _ in range(self.opt_num_epochs):
            # Update critic
            value_loss_val += self.update_value(states, returns)

            # Calculate log probabilities of selected actions
            if isinstance(self.policy_net, PolicyMOEWithPrev):
                prev_expert_indices = torch.cat(
                (torch.zeros(1, dtype=torch.int64).to(self.device), expert_indices[ind][:-1]),
                dim=0
                )
                prev_expert_indices = torch.nn.functional.one_hot(
                    prev_expert_indices, num_classes=self.cfg.num_experts
                ).float()
                log_probs = self.policy_net.get_log_prob(states[ind], prev_expert_indices, expert_indices[ind])
            elif expert_indices is not None:
                log_probs = self.policy_net.get_log_prob(states[ind], expert_indices[ind])
            else:
                log_probs = self.policy_net.get_log_prob(states[ind], actions[ind])

            # Calculate the loss as the negative log probability times the advantage
            policy_loss = -(log_probs * advantages[ind]).mean()

            # Backpropagate and update the policy network
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()
            policy_loss_val += policy_loss.item()

        mean_log_std = 0.0
        if hasattr(self.policy_net, "action_log_std"):
            mean_log_std = self.policy_net.action_log_std.mean().item()
        elif hasattr(self.policy_net, "log_std"):
            mean_log_std = self.policy_net.log_std.mean().item()

        return {
            "policy_loss": policy_loss_val / self.opt_num_epochs,
            "value_loss": value_loss_val / self.opt_num_epochs,
            "mean_log_std": mean_log_std
        }

    def update_params(self, batch) -> dict:
        """
        Perform parameter updates for both policy and value networks using the collected batch.

        Args:
            batch: A batch of collected experiences containing states, actions, rewards, masks, and exploration flags.

        Returns:
            dict: Dictionary containing training metrics and update time.
        """
        print("Updating parameters...")
        t0 = time.time()
        # Set the modules to training mode
        to_train(*self.update_modules)

        # Convert the batch to tensors
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        if hasattr(self.policy_net, "experts"):
            expert_indices = torch.from_numpy(batch.expert_indices).to(self.dtype).to(self.device)
        else:
            expert_indices = None

        # Compute value estimates for the states without gradient tracking
        with to_test(*self.update_modules):
            with torch.no_grad():
                values = self.value_net(states)

        # Estimate advantages and returns
        advantages, returns = estimate_advantages(
            rewards, masks, values, self.gamma, self.tau
        )

        metrics = self.update_policy(states, actions, returns, advantages, exps, expert_indices)
        metrics["update_time"] = time.time() - t0

        return metrics
