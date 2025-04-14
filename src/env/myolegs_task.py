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

from abc import abstractmethod
from typing import Optional, Tuple
import numpy as np
from src.env.myolegs_env import MyoLegsEnv


class MyoLegsTask(MyoLegsEnv):

    def __init__(self, cfg):
        super().__init__(cfg)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        # First reset humanoid, then reset task, then reset the simulation.
        "Resets at MyoLegsTask level"
        self.reset_task(options=options)
        return super().reset(seed=seed, options=options)

    @abstractmethod
    def get_task_obs_size(self):
        """Returns the size of the task-specific observations."""
        pass

    def get_obs_size(self):
        return self.get_self_obs_size() + self.get_task_obs_size()

    @abstractmethod
    def reset_task(self, options=None):
        """Resets task-specific state."""
        pass

    @abstractmethod
    def compute_task_obs(self):
        """Computes task-specific observations."""
        pass

    def render(self):
        if not self.headless:
            self.draw_task()
        return (
            super().render()
        )  # this may return an RGB image (if render_mode="rgb_array")

    @abstractmethod
    def draw_task(self):
        pass

    @abstractmethod
    def create_task_visualization(self):
        pass

    def compute_observations(self) -> np.ndarray:
        """
        Calls functions to compute proprioception and task observations and concatenates the returned arrays.
        """
        prop_obs = self.compute_proprioception()
        task_obs = self.compute_task_obs()
        return np.concatenate([prop_obs, task_obs])

    def create_viewer(self) -> None:
        """
        Adds the task visualization (e.g. motion imitation spheres) to the viewer.
        """
        super().create_viewer()
        self.create_task_visualization()
        self.draw_task()
