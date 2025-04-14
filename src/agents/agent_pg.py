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

    def update_value(self, states: torch.Tensor, returns: torch.Tensor) -> None:
        """
        Update the critic (value network) by minimizing the MSE between predicted values and returns.

        Args:
            states (torch.Tensor): Tensor of states.
            returns (torch.Tensor): Tensor of target returns.
        """
        for _ in range(self.value_opt_niter):
            values_pred = self.value_net(states)
            value_loss = (values_pred - returns).pow(2).mean()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        exps: torch.Tensor,
    ) -> None:
        """
        Update the policy network using advantage-weighted log probabilities.

        Args:
            states (torch.Tensor): Tensor of states.
            actions (torch.Tensor): Tensor of actions taken.
            returns (torch.Tensor): Tensor of target returns.
            advantages (torch.Tensor): Tensor of advantage estimates.
            exps (torch.Tensor): Tensor indicating exploration flags.
        """
        # use a2c by default
        ind = exps.nonzero().squeeze(1)
        for _ in range(self.opt_num_epochs):
            # Update critic
            self.update_value(states, returns)

            # Calculate log probabilities of selected actions
            log_probs = self.policy_net.get_log_prob(states[ind], actions[ind])

            # Calculate the loss as the negative log probability times the advantage
            policy_loss = -(log_probs * advantages[ind]).mean()

            # Backpropagate and update the policy network
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

    def update_params(self, batch) -> float:
        """
        Perform parameter updates for both policy and value networks using the collected batch.

        Args:
            batch: A batch of collected experiences containing states, actions, rewards, masks, and exploration flags.

        Returns:
            float: Time taken to perform the parameter updates.
        """
        t0 = time.time()
        # Set the modules to training mode
        to_train(*self.update_modules)

        # Convert the batch to tensors
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)

        # Compute value estimates for the states without gradient tracking
        with to_test(*self.update_modules):
            with torch.no_grad():
                values = self.value_net(states)

        # Estimate advantages and returns
        advantages, returns = estimate_advantages(
            rewards, masks, values, self.gamma, self.tau
        )

        self.update_policy(states, actions, returns, advantages, exps)

        return time.time() - t0
