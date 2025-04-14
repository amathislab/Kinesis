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

import math
from typing import List, Optional, Tuple
import numpy as np
import torch

from src.learning.learning_utils import to_test
from src.agents.agent_pg import AgentPG

import logging

logger = logging.getLogger(__name__)


class AgentPPO(AgentPG):

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        mini_batch_size: int = 64,
        use_mini_batch: bool = False,
        policy_grad_clip: Optional[List[Tuple[torch.nn.Module, float]]] = None,
        **kwargs
    ):
        """
        Initialize the PPO Agent.

        Args:
            clip_epsilon (float): Clipping parameter for PPO's surrogate objective.
            mini_batch_size (int): Size of mini-batches for stochastic gradient descent.
            use_mini_batch (bool): Whether to use mini-batch updates.
            policy_grad_clip (List[Tuple[torch.nn.Module, float]], optional):
                List of tuples containing networks and their max gradient norms for clipping.
            **kwargs: Additional parameters for the base AgentPG class.
        """
        super().__init__(**kwargs)

        # Initialize PPO parameters
        self.clip_epsilon = clip_epsilon
        self.mini_batch_size = mini_batch_size
        self.use_mini_batch = use_mini_batch
        self.policy_grad_clip = policy_grad_clip

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        exps: torch.Tensor,
    ) -> None:
        """
        Update the policy network using PPO's clipped surrogate objective.

        Args:
            states (torch.Tensor): Tensor of states.
            actions (torch.Tensor): Tensor of actions taken.
            returns (torch.Tensor): Tensor of target returns.
            advantages (torch.Tensor): Tensor of advantage estimates.
            exps (torch.Tensor): Tensor indicating exploration flags.
        """
        # Compute log proabilities of the actions under the current policy
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        for _ in range(self.opt_num_epochs):
            if self.use_mini_batch:
                perm = np.arange(states.shape[0])
                np.random.shuffle(perm)
                perm = torch.LongTensor(perm).to(self.device)

                states, actions, returns, advantages, fixed_log_probs, exps = (
                    states[perm].clone(),
                    actions[perm].clone(),
                    returns[perm].clone(),
                    advantages[perm].clone(),
                    fixed_log_probs[perm].clone(),
                    exps[perm].clone(),
                )

                optim_iter_num = int(math.floor(states.shape[0] / self.mini_batch_size))
                for i in range(optim_iter_num):
                    ind = slice(
                        i * self.mini_batch_size,
                        min((i + 1) * self.mini_batch_size, states.shape[0]),
                    )
                    (
                        states_b,
                        actions_b,
                        advantages_b,
                        returns_b,
                        fixed_log_probs_b,
                        exps_b,
                    ) = (
                        states[ind],
                        actions[ind],
                        advantages[ind],
                        returns[ind],
                        fixed_log_probs[ind],
                        exps[ind],
                    )
                    ind = exps_b.nonzero(as_tuple=False).squeeze(1)
                    self.update_value(states_b, returns_b)
                    surr_loss = self.ppo_loss(
                        states_b, actions_b, advantages_b, fixed_log_probs_b, ind
                    )
                    self.optimizer_policy.zero_grad()
                    surr_loss.backward()
                    self.clip_policy_grad()
                    self.optimizer_policy.step()
            else:

                ind = exps.nonzero(as_tuple=False).squeeze(1)
                self.update_value(states, returns)
                surr_loss = self.ppo_loss(
                    states, actions, advantages, fixed_log_probs, ind
                )
                self.optimizer_policy.zero_grad()
                surr_loss.backward()
                self.clip_policy_grad()
                self.optimizer_policy.step()

    def clip_policy_grad(self) -> None:
        """
        Clip gradients of the policy network to prevent exploding gradients.
        """
        if self.policy_grad_clip is not None:
            for net, max_norm in self.policy_grad_clip:
                total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)

    def ppo_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        fixed_log_probs: torch.Tensor,
        ind: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the PPO surrogate loss.

        Args:
            states (torch.Tensor): Tensor of states.
            actions (torch.Tensor): Tensor of actions taken.
            advantages (torch.Tensor): Tensor of advantage estimates.
            fixed_log_probs (torch.Tensor): Tensor of log probabilities under the old policy.
            ind (torch.Tensor): Tensor of indices indicating active exploration flags.

        Returns:
            torch.Tensor: Computed PPO surrogate loss.
        """
        log_probs = self.policy_net.get_log_prob(states[ind], actions[ind])
        ratio = torch.exp(log_probs - fixed_log_probs[ind])
        advantages = advantages[ind]
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * advantages
        )
        surr_loss = -torch.min(surr1, surr2).mean()

        return surr_loss
