# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# This file contains code adapted from:
#
# 1. PHC_MJX (https://github.com/ZhengyiLuo/PHC_MJX)

import os
import torch
import numpy as np
import logging

os.environ["OMP_NUM_THREADS"] = "1"

from src.agents.agent_humanoid import AgentHumanoid
from src.learning.learning_utils import to_test, to_cpu
from src.env.myolegs_im import MyoLegsIm

logger = logging.getLogger(__name__)


class AgentIM(AgentHumanoid):
    """
    AgentIM is a specialized reinforcement learning agent for humanoid environments,
    extending AgentHumanoid with specific functionalities for the MyoLegsIm environment.
    """
    
    def __init__(self, cfg, dtype, device, training: bool = True, checkpoint_epoch: int = 0):
        """
        Initialize the AgentIM with configurations and set up necessary components.

        Args:
            cfg: Configuration object containing hyperparameters and settings.
            dtype: Data type for tensors (e.g., torch.float32).
            device: Device for computations (e.g., 'cuda' or 'cpu').
            training (bool, optional): Flag indicating if the agent is in training mode.
            checkpoint_epoch (int, optional): Epoch number from which to load the checkpoint.
        """
        super().__init__(cfg, dtype, device, training, checkpoint_epoch)

    def get_full_state_weights(self) -> dict:
        """
        Extends the state dictionary with termination history for checkpointing.

        Returns:
            dict: The state dictionary including termination history.
        """
        state = super().get_full_state_weights()
        return state
    
    def set_full_state_weights(self, state) -> None:
        """
        Loads the state dictionary.

        Args:
            state (dict): The state dictionary including termination history.
        """
        super().set_full_state_weights(state)
        
    
    def pre_epoch(self) -> None:
        """
        Performs operations before each training epoch, such as resampling motions.
        """
        if (self.epoch > 1) and self.epoch % self.cfg.env.resampling_interval == 1: # + 1 to evade the evaluations. 
            self.env.sample_motions()
        return super().pre_epoch()
    
    def setup_env(self):
        """
        Initializes the MyoLegsIm environment based on the configuration.
        """
        self.env = MyoLegsIm(self.cfg)
        logger.info("MyoLegsIm environment initialized.")

    def eval_policy(self, epoch: int = 0, dump: bool = False, runs = None) -> float:
        """
        Evaluates the current policy by running multiple episodes and computing success rates.

        Args:
            epoch (int, optional): Current epoch number for logging and checkpointing.
            dump (bool, optional): Flag indicating whether to dump evaluation results.

        Returns:
            float: The success rate of the policy.
        """
        logger.info("Starting policy evaluation.")
        res_dict_acc = {}
        self.env.start_eval(im_eval = True)

        # Set networks to evaluation mode
        to_test(*self.sample_modules)

        success_dict = {}
        mpjpe_dict = {}
        frame_coverage_dict = {}

        if runs is not None:
            run_ctr = 0

        with to_cpu(*self.sample_modules), torch.no_grad():
            for run_idx in self.env.forward_motions():
                success = False
                for attempt in range(1):
                    result, mpjpe, frame_coverage = self.eval_single_thread()
                    if result is True:
                        success = True
                        logger.info(f"Run {run_idx}: Success on attempt {attempt + 1}. MPJPE: {mpjpe * 1000:.5f}, Frame Coverage: {frame_coverage * 100:.5f}")
                    else:
                        success = False
                        logger.info(f"Run {run_idx}: Failure on attempt {attempt + 1}. MPJPE: {mpjpe * 1000:.5f}, Frame Coverage: {frame_coverage[0] * 100:.5f}")
                    success_dict[run_idx] = success
                    mpjpe_dict[run_idx] = mpjpe
                    frame_coverage_dict[run_idx] = frame_coverage
                if runs is not None:
                    run_ctr += 1
                    if run_ctr >= runs:
                        break
                
        success_rate = np.mean(list(success_dict.values()))
        mean_mpjpe = np.mean(list(mpjpe_dict.values()))
        mean_frame_coverage = np.mean(list(frame_coverage_dict.values()))
        failed_keys = [k for k, v in success_dict.items() if not v]
        success_keys = [k for k, v in success_dict.items() if v]
        print(f"Success Rate: {success_rate * 100:.5f}")
        print("Mean MPJPE: ", mean_mpjpe * 1000)
        print("Mean frame coverage: ", mean_frame_coverage * 100)

        # save failed keys
        if dump:
            os.makedirs("data/dumps", exist_ok=True)
            failed_keys = np.array(failed_keys)
            np.save(f"data/dumps/failed_keys_{self.cfg.epoch}.npy", failed_keys)

        if self.env.recording_biomechanics:
            breakpoint()
            print("Saving recorded biomechanics data.")
            

        return mpjpe_dict, success_rate

    
    def eval_single_thread(self) -> bool:
        """
        Evaluates the policy in a single thread by running an episode.

        Returns:
            bool: True if the episode terminated successfully, False otherwise.
        """
        with to_cpu(*self.sample_modules), torch.no_grad():
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
                    return not terminated, self.env.mpjpe_value, self.env.frame_coverage
                state = next_state

        # If the loop exits without termination, consider it a failure
        return False, self.env.mpjpe, self.env.frame_coverage
            
            
    def run_policy(self, epoch: int = 0, dump: bool = False) -> dict:
        """
        Runs the trained policy in the environment.

        Args:
            epoch (int, optional): Current epoch number.
            dump (bool, optional): Flag indicating whether to dump run results.

        Returns:
            dict: Run metrics.
        """
        self.env.start_eval(im_eval = False)
        return super().run_policy(epoch, dump)