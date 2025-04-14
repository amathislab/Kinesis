# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.

import numpy as np
from typing import Tuple

from src.env.myolegs_pointgoal import MyoLegsPointGoal

from scipy.spatial.transform import Rotation as sRot

import logging

logger = logging.getLogger(__name__)

class MyoLegsDirectional(MyoLegsPointGoal):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.goal_pos = np.zeros(3)
        self.previous_tracking_distance = None

        self.direction = 0
        self.stop = False
        self.speed = np.array([-1.265, 0, 0])

    def compute_reset(self) -> Tuple[bool, bool]:
        """
        Determines whether the task should be reset based on termination and truncation conditions.

        Termination means failure, while truncation means success.

        Returns:
            tuple:
                - terminated (bool): If the agent falls before the episode times out, or if the episode times out but the agent is not close enough to the goal.
                - truncated (bool): If the agent is standing close enough to the goal when the episode times out.
        """
        body_pos = self.get_body_xpos()
        root_pos = body_pos[0]
        self.goal_pos = root_pos if self.stop else root_pos + sRot.from_euler("z", self.direction).apply(self.speed)

        fall_terminated = self.proprioception["root_height"] < 0.7
        timetout_terminated = self.cur_t >= 1500000
        truncated = timetout_terminated and np.linalg.norm(self.get_body_xpos()[0, :2] - self.goal_pos[:2]) < 0.1
        terminated = fall_terminated or (timetout_terminated and not truncated)


        return terminated, truncated
    
    def reset_task(self, options=None) -> None:
        """
        Resets the task to an initial state by reinitializing key motion parameters.

        The function selects a random start time for the reference motion that the agent
        uses to initialize its position.

        Args:
            options (dict, optional): Additional options for task resetting. Defaults to None.
        """
        self.goal_pos = np.zeros(3)
        self.previous_tracking_distance = None

        self.direction = 0
        self.speed = np.array([-1.265, 0, 0])

        self.stop = True

        return
    
    def key_callback(self, keycode):
        super().key_callback(keycode)
        if keycode == 321:
            # Point the goal to north-east
            self.stop = False
            self.direction = np.pi / 4
        elif keycode == 322:
            # Point the goal to north
            self.stop = False
            self.direction = np.pi / 2
        elif keycode == 323:
            # Point the goal to north-west
            self.stop = False
            self.direction = 3 * np.pi / 4
        elif keycode == 326:
            # Point the goal to west
            self.stop = False
            self.direction = np.pi
        elif keycode == 329:
            # Point the goal to south-west
            self.stop = False
            self.direction = 5 * np.pi / 4
        elif keycode == 328:
            # Point the goal to south
            self.stop = False
            self.direction = 3 * np.pi / 2
        elif keycode == 327:
            # Point the goal to south-east
            self.stop = False
            self.direction = 7 * np.pi / 4
        elif keycode == 324:
            # Point the goal to east
            self.stop = False
            self.direction = 0
        elif keycode == 325:
            # Stand still
            self.stop = True
        elif keycode == 333:
            # Decrease the speed
            self.speed[0] = self.speed[0] + 0.1 if np.abs(self.speed[0]) > 0.1 else self.speed[0]
        elif keycode == 334:
            # Increase the speed
            self.speed[0] = self.speed[0] - 0.1 if np.abs(self.speed[0]) < 3.0 else self.speed[0]

