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

from src.learning.policy_moe import PolicyMOEWithPrev
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
        expert_indices: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Update the policy network using PPO's clipped surrogate objective.

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
        print("Updating policy...")
        # Compute log proabilities of the actions under the current policy
        with to_test(*self.update_modules):
            with torch.no_grad():
                if isinstance(self.policy_net, PolicyMOEWithPrev):
                    # For PolicyMOEWithPrev, we need to pass the previous expert index
                    
                    # Roll the expert indices forward one step and add a zero for the first step
                    prev_expert_indices = torch.cat(
                        (torch.zeros(1, dtype=torch.int64).to(self.device), expert_indices[:-1].long().clamp(0, self.cfg.num_experts - 1)),
                        dim=0
                    )
                    # Transform to one-hot encoding
                    prev_expert_indices = torch.nn.functional.one_hot(
                        prev_expert_indices, num_classes=self.cfg.num_experts
                    ).float()

                # Compute fixed_log_probs in chunks to avoid OOM
                chunk_size = 4096
                fixed_log_probs_list = []
                num_samples = states.shape[0]
                
                for i in range(0, num_samples, chunk_size):
                    end = min(i + chunk_size, num_samples)
                    states_chunk = states[i:end]
                    actions_chunk = actions[i:end]
                    
                    if isinstance(self.policy_net, PolicyMOEWithPrev):
                        prev_expert_indices_chunk = prev_expert_indices[i:end]
                        expert_indices_chunk = expert_indices[i:end]
                        fixed_log_probs_chunk = self.policy_net.get_log_prob(states_chunk, prev_expert_indices_chunk, expert_indices_chunk)
                    elif expert_indices is not None:
                        expert_indices_chunk = expert_indices[i:end]
                        fixed_log_probs_chunk = self.policy_net.get_log_prob(states_chunk, expert_indices_chunk)
                    else:
                        fixed_log_probs_chunk = self.policy_net.get_log_prob(states_chunk, actions_chunk)
                    
                    fixed_log_probs_list.append(fixed_log_probs_chunk)
                
                fixed_log_probs = torch.cat(fixed_log_probs_list, dim=0)

        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        clip_fraction_sum = 0.0
        n_updates = 0

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
                    v_loss = self.update_value(states_b, returns_b)
                    surr_loss, clip_frac = self.ppo_loss(
                        states_b, actions_b, advantages_b, fixed_log_probs_b, ind
                    )
                    self.optimizer_policy.zero_grad()
                    surr_loss.backward()
                    self.clip_policy_grad()
                    self.optimizer_policy.step()

                    policy_loss_sum += surr_loss.item()
                    value_loss_sum += v_loss
                    clip_fraction_sum += clip_frac
                    n_updates += 1
            else:

                ind = exps.nonzero(as_tuple=False).squeeze(1)
                v_loss = self.update_value(states, returns)
                
                # Gradient accumulation to avoid OOM
                self.optimizer_policy.zero_grad()
                
                chunk_size = 4096
                policy_loss_accum = 0.0
                clip_frac_accum = 0.0
                
                num_samples = ind.shape[0]
                if num_samples > 0:
                    for i in range(0, num_samples, chunk_size):
                        end = min(i + chunk_size, num_samples)
                        ind_chunk = ind[i:end]
                        
                        surr_loss_chunk, clip_frac_chunk = self.ppo_loss(
                            states, actions, advantages, fixed_log_probs, ind_chunk, expert_indices
                        )
                        
                        # Scale loss by chunk size / total size for correct mean gradient
                        loss_weight = (end - i) / num_samples
                        (surr_loss_chunk * loss_weight).backward()
                        
                        policy_loss_accum += surr_loss_chunk.item() * loss_weight
                        clip_frac_accum += clip_frac_chunk * loss_weight

                    self.clip_policy_grad()
                    self.optimizer_policy.step()

                policy_loss_sum += policy_loss_accum
                value_loss_sum += v_loss
                clip_fraction_sum += clip_frac_accum
                n_updates += 1
        
        mean_log_std = 0.0
        if hasattr(self.policy_net, "action_log_std"):
            mean_log_std = self.policy_net.action_log_std.mean().item()
        elif hasattr(self.policy_net, "log_std"):
            mean_log_std = self.policy_net.log_std.mean().item()

        return {
            "policy_loss": policy_loss_sum / n_updates if n_updates > 0 else 0.0,
            "value_loss": value_loss_sum / n_updates if n_updates > 0 else 0.0,
            "clip_fraction": clip_fraction_sum / n_updates if n_updates > 0 else 0.0,
            "mean_log_std": mean_log_std
        }

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
        expert_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Calculate the PPO surrogate loss.

        Args:
            states (torch.Tensor): Tensor of states.
            actions (torch.Tensor): Tensor of actions taken.
            advantages (torch.Tensor): Tensor of advantage estimates.
            fixed_log_probs (torch.Tensor): Tensor of log probabilities under the old policy.
            ind (torch.Tensor): Tensor of indices indicating active exploration flags.
            expert_indices (Optional[torch.Tensor]): Tensor of expert indices for actions, if applicable.

        Returns:
            Tuple[torch.Tensor, float]: Computed PPO surrogate loss and clip fraction.
        """
        if isinstance(self.policy_net, PolicyMOEWithPrev):
            # For PolicyMOEWithPrev, we need to pass the previous expert index
            prev_expert_indices = torch.cat(
                (torch.zeros(1, dtype=torch.int64).to(self.device), expert_indices[ind][:-1]),
                dim=0
            )
            prev_expert_indices = torch.nn.functional.one_hot(
                prev_expert_indices.long().clamp(0, self.cfg.num_experts - 1), num_classes=self.cfg.num_experts
            ).float()
            log_probs = self.policy_net.get_log_prob(states[ind], prev_expert_indices, expert_indices[ind])
        elif expert_indices is not None:
            log_probs = self.policy_net.get_log_prob(states[ind], expert_indices[ind])
        else:
            log_probs = self.policy_net.get_log_prob(states[ind], actions[ind])
        ratio = torch.exp(log_probs - fixed_log_probs[ind])
        advantages = advantages[ind]
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * advantages
        )
        surr_loss = -torch.min(surr1, surr2).mean()

        clipped = ratio.gt(1.0 + self.clip_epsilon) | ratio.lt(1.0 - self.clip_epsilon)
        clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        return surr_loss, clip_fraction
