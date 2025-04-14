# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# This file contains code adapted from:
#
# 1. PHC_MJX (https://github.com/ZhengyiLuo/PHC_MJX)

from abc import ABC, abstractmethod
import time
import os
import torch
import wandb
import numpy as np
import psutil

os.environ["OMP_NUM_THREADS"] = "1"

from src.agents.agent_ppo import AgentPPO
from src.learning.policy_gaussian import PolicyGaussian
from src.learning.policy_lattice import PolicyLattice
from src.learning.policy_moe import PolicyMOE
from src.learning.critic import Value
from src.learning.mlp import MLP
from src.learning.learning_utils import to_device, to_cpu, get_optimizer

import logging

logger = logging.getLogger(__name__)


class AgentHumanoid(AgentPPO, ABC):
    """
    Contains functions to load, save, train, and run policies for the humanoid agent.
    """

    def __init__(self, cfg, dtype, device, training=True, checkpoint_epoch=0):
        """
        Initialize the AgentHumanoid with configurations and set up necessary components.

        Args:
            cfg: Configuration object containing hyperparameters and settings.
            dtype: Data type for tensors (e.g., torch.float32).
            device: Device for computations (e.g., 'cuda' or 'cpu').
            training (bool, optional): Flag indicating if the agent is in training mode.
            checkpoint_epoch (int, optional): Epoch number from which to load the checkpoint.
        """
        self.cfg = cfg
        self.cc_cfg = cfg
        self.device = device
        self.dtype = dtype
        self.training = training
        self.max_freq = 50

        self.setup_vars()
        self.setup_env()
        self.setup_policy()
        self.setup_value()
        self.setup_optimizer()
        self.seed(cfg.seed)
        self.print_config()
        self.load_checkpoint(checkpoint_epoch)

        super().__init__(
            env=self.env,
            dtype=dtype,
            device=device,
            mean_action=cfg.test,
            headless=not cfg.headless,
            num_threads=cfg.run.num_threads,
            policy_net=self.policy_net,
            value_net=self.value_net,
            optimizer_policy=self.optimizer_policy,
            optimizer_value=self.optimizer_value,
            opt_num_epochs=cfg.learning.opt_num_epochs,
            gamma=cfg.learning.gamma,
            tau=cfg.learning.tau,
            clip_epsilon=cfg.learning.clip_epsilon,
            policy_grad_clip=[(self.policy_net, cfg.learning.policy_grad_clip)],
            use_mini_batch=False,
            mini_batch_size=0,
            clip_obs=cfg.learning.clip_obs,
            clip_obs_range=cfg.learning.clip_obs_range,
            clip_actions=cfg.env.clip_actions,
        )

    def setup_vars(self):
        """
        Initialize basic variables required by the agent.
        """
        self.epoch = 0

    def print_config(self) -> None:
        """
        Log the current configuration and relevant settings.
        """
        logger.info(
            "==========================Agent Parameters==========================="
        )
        logger.info(f"State_dim: {self.state_dim}")
        logger.info("============================================================")

    @abstractmethod
    def setup_env(self) -> None:
        raise NotImplementedError

    def setup_policy(self) -> None:
        """
        Initialize the policy network based on the configuration.
        Supports various policy architectures.
        """
        self.state_dim = state_dim = self.env.observation_space.shape[0]
        self.action_dim = action_dim = self.env.action_space.shape[0]

        if self.cfg.learning.actor_type == "gauss":
            self.policy_net = PolicyGaussian(
                self.cfg, action_dim=action_dim, state_dim=state_dim
            )
        elif self.cfg.learning.actor_type == "lattice":
            self.policy_net = PolicyLattice(
                self.cfg, action_dim=action_dim, latent_dim=512, state_dim=state_dim
            )
        elif self.cfg.learning.actor_type == "moe":
            self.policy_net = PolicyMOE(
                self.cfg, action_dim=action_dim, state_dim=state_dim
            )
        elif self.cfg.learning.actor_type == "moe_finetune":
            self.policy_net = PolicyMOE(
                self.cfg, action_dim=action_dim, state_dim=state_dim, freeze=False
            )

        to_device(self.device, self.policy_net)

    def setup_value(self) -> None:
        """
        Initialize the value network using an MLP architecture.
        """
        state_dim = self.env.observation_space.shape[0]
        self.value_net = Value(
            MLP(
                state_dim, self.cfg.learning.mlp.units, self.cfg.learning.mlp.activation
            )
        )
        to_device(self.device, self.value_net)

    def setup_optimizer(self) -> None:
        """
        Set up optimizers for the policy and value networks.
        """
        self.optimizer_policy = get_optimizer(
            self.policy_net,
            self.cfg.learning.policy_lr,
            self.cfg.learning.policy_weightdecay,
            self.cfg.learning.policy_optimizer,
        )
        self.optimizer_value = get_optimizer(
            self.value_net,
            self.cfg.learning.value_lr,
            self.cfg.learning.value_weightdecay,
            self.cfg.learning.value_optimizer,
        )

    def get_nn_weights(self) -> dict:
        """
        Retrieve the state dictionaries of the policy and value networks.

        Returns:
            dict: Contains 'policy' and 'value' state dictionaries.
        """
        state = {}
        state["policy"] = self.policy_net.state_dict()
        state["value"] = self.value_net.state_dict()
        return state

    def set_nn_weights(self, weights) -> None:
        """
        Load state dictionaries into the policy and value networks.

        Args:
            weights (dict): Contains 'policy' and 'value' state dictionaries.
        """
        self.policy_net.load_state_dict(weights["policy"])
        self.value_net.load_state_dict(weights["value"])

    def get_full_state_weights(self) -> dict:
        """
        Retrieve the full state, including network weights and optimizer states.

        Returns:
            dict: Comprehensive state including networks, optimizers, epoch, and frame count.
        """
        return {
            "policy": self.policy_net.state_dict(),
            "value": self.value_net.state_dict(),
            "epoch": self.epoch,
            "optimizer_policy": self.optimizer_policy.state_dict(),
            "optimizer_value": self.optimizer_value.state_dict(),
            "frame": self.num_steps,
        }

    def set_full_state_weights(self, state):
        """
        Load the full state, including network weights and optimizer states.

        Args:
            state (dict): Comprehensive state including networks, optimizers, epoch, and frame count.
        """
        self.set_nn_weights(state)
        self.epoch = state["epoch"]
        self.optimizer_value.load_state_dict(state["optimizer_value"])
        self.optimizer_policy.load_state_dict(state["optimizer_policy"])
        self.num_steps = state.get("frame", 0)
        print(
            f"==============================Loaded checkpoint model: Epoch {self.epoch}=============================="
        )

    def save_checkpoint(self) -> None:
        """
        Save the current state as a checkpoint.
        """
        torch.save(
            self.get_full_state_weights(),
            f"{self.cfg.output_dir}/model_{self.epoch:08d}.pth",
        )
        print(
            f"==============================Saved checkpoint model: Epoch {self.epoch}=============================="
        )

    def save_curr(self) -> None:
        """
        Save the current state as the latest model.
        """
        torch.save(self.get_full_state_weights(), f"{self.cfg.output_dir}/model.pth")
        print(
            f"==============================Saved current model: Epoch {self.epoch}=============================="
        )

    def load_checkpoint(self, epoch):
        """
        Load a checkpoint based on the specified epoch.

        Args:
            epoch (int): Epoch number to load. -1 loads the latest model.
        """
        if epoch == -1:
            checkpoint_path = os.path.join(self.cfg.output_dir, "model.pth")
            if os.path.exists(checkpoint_path):
                state = torch.load(checkpoint_path, map_location=self.device)
                self.set_full_state_weights(state)
                logger.info(f"Loaded latest checkpoint from {checkpoint_path}")
            else:
                logger.warning(f"No checkpoint found at {checkpoint_path}")
        elif epoch > 0:
            checkpoint_path = os.path.join(
                self.cfg.output_dir, f"model_{epoch:08d}.pth"
            )
            if os.path.exists(checkpoint_path):
                state = torch.load(checkpoint_path, map_location=self.device)
                self.set_full_state_weights(state)
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
            else:
                logger.warning(f"No checkpoint found at {checkpoint_path}")
        else:
            logger.info("Starting model from scratch.")

        # Load motions
        if self.cfg.run.im_eval == False:
            self.env.sample_motions()

        to_device(self.device, self.policy_net, self.value_net)

    def pre_epoch(self) -> None:
        pass

    def after_epoch(self) -> None:
        pass

    def log_train(self, info) -> None:
        """
        Log training metrics and information.

        Args:
            info (dict): Contains logging information such as loggers, timings, etc.
        """
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / 1024 / 1024 / 1024
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024 * 100
        loggers = info["loggers"]

        self.num_steps += loggers.num_steps
        reward_str = " ".join([f"{v:.3f}" for k, v in loggers.info_dict.items()])

        log_str = f"Ep: {self.epoch} \t {self.cfg.exp_name} T_s {info['T_sample']:.2f}  T_u { info['T_update']:.2f} \t eps_R_avg {loggers.avg_episode_reward:.4f} R_avg {loggers.avg_reward:.4f} R_range ({loggers.min_reward:.4f}, {loggers.max_reward:.4f}) [{reward_str}] \t num_s { self.num_steps} eps_len {loggers.avg_episode_len:.2f}"

        logger.info(log_str)

        if not self.cfg.no_log:
            wandb.log(
                data={
                    "avg_episode_reward": loggers.avg_episode_reward,
                    "eps_len": loggers.avg_episode_len,
                    "avg_rwd": loggers.avg_reward,
                    "reward_raw": loggers.info_dict,
                    "cpu_mem": cpu_mem,
                    "gpu_mem": gpu_mem,
                    "t_sample": info["T_sample"],
                    "t_update": info["T_update"],
                },
                step=self.epoch,
            )

            if "log_eval" in info:
                wandb.log(data=info["log_eval"], step=self.epoch)

        return loggers

    def optimize_policy(self, save_model: bool = True) -> None:
        """
        Execute the main training loop, optimizing the policy and value networks.

        Args:
            save_model (bool, optional): Flag indicating whether to save model checkpoints.
        """
        eps_len_list = []
        starting_epoch = self.epoch
        for _ in range(starting_epoch, self.cfg.learning.max_epoch):
            t0 = time.time()
            self.pre_epoch()
            batch, loggers = self.sample(self.cfg.learning.min_batch_size)

            # Update the policy and value networks
            t1 = time.time()
            self.update_params(batch)

            self.epoch += 1

            if save_model and (self.epoch) % self.cfg.learning.save_frequency == 0:
                self.save_checkpoint()
                # log_eval = self.eval_policy()
                # info["log_eval"] = log_eval
            elif (
                save_model and (self.epoch) % self.cfg.learning.save_curr_frequency == 0
            ):
                self.save_curr()
            t2 = time.time()

            info = {
                "loggers": loggers,
                "T_sample": t1 - t0,
                "T_update": t2 - t1,
                "T_total": t2 - t0,
            }

            loggers = self.log_train(info)
            eps_len_list.append(loggers.avg_episode_len)
            self.after_epoch()

        return eps_len_list

    def eval_policy(self, epoch: int = 0, dump: bool = False) -> dict:
        """
        Evaluate the current policy. Placeholder implementation.

        Args:
            epoch (int, optional): Current epoch number.
            dump (bool, optional): Flag indicating whether to dump evaluation results.

        Returns:
            dict: Evaluation metrics.
        """
        res_dicts = {}
        return res_dicts

    def run_policy(self, epoch: int = 0, dump: bool = False) -> dict:
        """
        Run the trained policy in the environment indefinitely until episodes terminate.

        Args:
            epoch (int, optional): Current epoch number.
            dump (bool, optional): Flag indicating whether to dump run results.

        Returns:
            dict: Run metrics.
        """
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                while True:
                    obs_dict, info = self.env.reset()
                    state = self.preprocess_obs(obs_dict)
                    for t in range(10000):
                        actions = self.policy_net.select_action(
                            torch.from_numpy(state).to(self.dtype), True
                        )[0].numpy()

                        next_obs, reward, terminated, truncated, info = self.env.step(
                            self.preprocess_actions(actions)
                        )
                        next_state = self.preprocess_obs(next_obs)
                        done = terminated or truncated

                        if done:
                            break
                        state = next_state
        res_dicts = {}
        return res_dicts

    def seed(self, seed):
        """
        Seed the random number generators for reproducibility.

        Args:
            seed (int): Seed value.
        """
        cfg, device, dtype = self.cfg, self.device, self.dtype
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.env.seed(seed)
