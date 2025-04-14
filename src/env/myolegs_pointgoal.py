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

import numpy as np
from typing import Tuple
import mujoco

from src.env.myolegs_im import MyoLegsIm, compute_imitation_observations
from src.env.myolegs_env import action_to_target_length, target_length_to_activation
from src.utils.visual_capsule import add_visual_capsule

from scipy.spatial.transform import Rotation as sRot

import logging

logger = logging.getLogger(__name__)

class MyoLegsPointGoal(MyoLegsIm):
    """
    Implements two high-level control tasks for the KINESIS(MyoLeg) framework:
    1. Target Goal Reaching
    2. Directional Control

    Attributes:
        goal_pos (np.ndarray): 
            A numpy array of shape (3,) representing the current goal position in 3D space. 
            The agent will always try to reach this goal position with its root (pelvis).

        previous_tracking_distance (float or None): 
            The previous distance to the goal used for tracking progress. 
            This variable is used to calculate the tracking reward (see below).
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.goal_pos = np.zeros(3)
        self.previous_tracking_distance = None

        if self.cfg.run.test == True:
            self.results_list = []

    def create_task_visualization(self) -> None:
        """
        Creates a visual representation of the task in the viewer.

        This function adds a visual capsule to the user scene of the viewer 
        if the viewer is initialized. The capsule serves as a marker for the 
        goal position.
        """
        if self.viewer is not None:
            add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))

    def draw_task(self) -> None:
        """
        Updates the visual representation of the task.

        This function updates the position of the visual object in the user 
        scene to match the current goal position.
        """
        if self.viewer is not None and self.goal_pos is not None:
            self.viewer.user_scn.geoms[0].pos = self.goal_pos
    
    def compute_tracking_distance(self) -> float:
        """
        Computes the Euclidean distance between the root position and the goal position.

        The distance is calculated in the 2D plane (ignoring the z-axis).

        Returns:
            float: The tracking distance.
        """
        body_pos = self.get_body_xpos()
        root_pos = body_pos[0]
        tracking_distance = np.linalg.norm(root_pos[:2] - self.goal_pos[:2])
        return tracking_distance
    
    def init_myolegs(self) -> None:
        """
        Initializes the MyoLegs environment.

        This method extends the superclass `init_myolegs` function, and 
        additionally "warm starts" the task by computing the 
        current tracking distance.
        """
        super().init_myolegs()
        self.previous_tracking_distance = self.compute_tracking_distance()
        return
    
    def initialize_motion_state(self) -> None:
        """
        Retrieves motion data from the motion library and sets the initial pose.
        """
        if self.cfg.run.test == False:
            super().initialize_motion_state()

        else:
            self.mj_data.qpos[:] = 0
            self.mj_data.qvel[:] = 0
            self.mj_data.qpos[2] = 0.94
            self.mj_data.qpos[3:7] = np.array([0.5, 0.5, 0.5, 0.5])

            initial_rot = sRot.from_euler("XYZ", [np.pi / 2, 0, -np.pi / 2])
            ref_qpos = np.array(
                [
                    0.70731837,
                    0.6811051,
                    0.9361768,
                    0.50487715,
                    0.4650055,
                    -0.50120455,
                    -0.5269373,
                ]
            ).astype(np.float32)
            self.mj_data.qpos[:3] = ref_qpos[:3]
            self.mj_data.qpos[3:7] = (initial_rot * sRot.from_quat(ref_qpos[3:7])).as_quat()

            self.initial_pose = np.array(
                [
                    0.57133061,
                    -1.21943974,
                    0.93766457,
                    0.84073663,
                    0.02309985,
                    -0.02534906,
                    0.54035704,
                    0.05718979,
                    -0.08046306,
                    -0.36137755,
                    0.00613998,
                    0.00156276,
                    0.34577042,
                    0.03181017,
                    0.15059691,
                    0.23338945,
                    0.22419988,
                    0.03759186,
                    -0.01772005,
                    0.01923135,
                    -0.11780726,
                    0.07734922,
                    -0.03445803,
                    -0.33847641,
                    -0.00639306,
                    0.00154999,
                    0.35934485,
                    0.0327628,
                    -0.16855011,
                    0.20909414,
                    0.2539858,
                    0.03759526,
                    -0.01772008,
                    0.01923106,
                    -0.11780739,
                ]
            ).astype(np.float32)
            self.mj_data.qpos[7:] = self.initial_pose[7:]

            # Set up velocity
            self.mj_data.qvel[:6] = 0
            self.mj_data.qvel[6:] = 0

            # Run kinematics
            mujoco.mj_kinematics(self.mj_model, self.mj_data)

    def compute_reset(self) -> Tuple[bool, bool]:
        """
        Determines whether the task should be reset based on termination and truncation conditions.

        Termination means failure, while truncation means success.

        Returns:
            tuple:
                - terminated (bool): If the agent falls before the episode times out, or if the episode times out but the agent is not close enough to the goal.
                - truncated (bool): If the agent is standing close enough to the goal when the episode times out.
        """
        fall_terminated = self.proprioception["root_height"] < 0.7
        timetout_terminated = self.cur_t >= 150
        truncated = timetout_terminated and np.linalg.norm(self.get_body_xpos()[0, :2] - self.goal_pos[:2]) < 0.1
        terminated = fall_terminated or (timetout_terminated and not truncated)

        if self.cfg.run.test == True:
            if truncated:
                self.results_list.append(True)
            elif terminated:
                self.results_list.append(False)

        return terminated, truncated
    
    def reset_task(self, options=None) -> None:
        """
        Resets the task to an initial state by reinitializing key motion parameters.

        The function selects a random start time for the reference motion that the agent
        uses to initialize its position.

        Args:
            options (dict, optional): Additional options for task resetting. Defaults to None.
        """
        self.goal_pos[:2] = np.random.uniform(-2, 2, size=2)
        self.goal_pos[2] = 0.94

        return
    
    def compute_task_obs(self) -> np.ndarray:
        """
        Computes task-specific observations used for imitation learning or control.

        This function calculates observations based on the current state of the body and 
        its relation to the goal position. It includes positional and velocity differences, 
        local reference positions, and biomechanical data (if enabled).

        The observations are structured to be useful for downstream tasks such as 
        imitation learning or reinforcement learning.

        Returns:
            np.ndarray: A concatenated array of task observations, including positional 
            differences, velocity differences, and local reference positions.

        Notes:
            - The reason we overload this function from `MyoLegsIm` is to set every
            reference position except the root to the current position of the agent.
            - If biomechanics recording is enabled, updates the list of foot contact states 
            and joint positions for analysis.
            - The returned observations are computed in a normalized and relative format 
            to ensure consistent scale and alignment.

        Observation Structure:
            - `diff_local_body_pos`: Difference in local body positions.
            - `diff_local_vel`: Difference in local body velocities.
            - `local_ref_body_pos`: Local reference body positions.

        Updates:
            - If `self.recording_biomechanics` is True, the function:
            - Updates `self.feet` with the current state of foot contacts.
            - Appends the current joint positions to `self.joint_pos`.

        """
        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]

        root_rot = body_rot[:, 0]
        root_pos = body_pos[:, 0]

        body_pos_subset = body_pos[..., self.track_bodies_id, :]

        ref_pos_subset = body_pos_subset
        ref_pos_subset[..., 0, :] = self.goal_pos

        body_vel = self.get_body_linear_vel()[None,]
        body_vel_subset = body_vel[..., self.track_bodies_id, :]

        zeroed_task_obs = compute_imitation_observations(
            root_pos,
            root_rot,
            body_pos_subset,
            body_vel_subset,
            ref_pos_subset,
            body_vel_subset,
            time_steps=1,
        )

        task_obs = {}
        task_obs["diff_local_body_pos"] = zeroed_task_obs["diff_local_body_pos"]
        task_obs["diff_local_vel"] = zeroed_task_obs["diff_local_vel"]
        task_obs["local_ref_body_pos"] = zeroed_task_obs["local_ref_body_pos"]

        # Update feet contacts
        if self.recording_biomechanics:
            feet_contacts = self.proprioception["feet_contacts"]
            planted_feet = -1
            if feet_contacts[0] > 0 or feet_contacts[1] > 0:
                planted_feet = 1
            if feet_contacts[2] > 0 or feet_contacts[3] > 0:
                planted_feet = 0
            if (feet_contacts[0] > 0 or feet_contacts[1] > 0) and (feet_contacts[2] > 0 or feet_contacts[3] > 0):
                planted_feet = 2
            self.feet.append(planted_feet)
            self.joint_pos.append(self.get_qpos().copy())
            self.joint_vel.append(self.get_qvel().copy())
            self.body_pos.append(body_pos.copy())
            self.body_rot.append(body_rot.copy())
            self.body_vel.append(body_vel.copy())
            self.muscle_forces.append(self.get_muscle_force().copy())
            self.muscle_controls.append(self.mj_data.ctrl.copy())

        return np.concatenate(
            [v.ravel() for v in task_obs.values()], axis=0, dtype=self.dtype
        )
    
    def compute_reward(self, action):
        """
        Computes the reward for the current timestep based on task performance metrics.

        The reward is calculated as a weighted combination of several components (see below).

        Args:
            action (np.ndarray): The action applied at the current timestep.

        Returns:
            float: The computed reward for the current timestep.

        Reward Components:
            - `tracking_reward`: Proportional to the improvement in distance to the goal, scaled by a factor of 20.
            - `energy_reward`: Average power usage (negative reward for excessive energy consumption).
            - `upright_reward`: Encourages an upright posture based on orientation.
            - `success_reward`: Fixed bonus (100) for reaching the goal within a threshold distance (0.1 units).

        Updates:
            - `self.previous_tracking_distance`: Stores the current tracking distance for use in the next step.
            - `self.curr_power_usage`: Clears the current power usage data.
            - `self.reward_info`: Dictionary storing individual reward components for analysis.

        Notes:
            - The final reward is a weighted combination of components, with weights specified in `self.reward_specs`.
        """
        body_pos = self.get_body_xpos()
        root_pos = body_pos[0]

        current_tracking_distance = np.linalg.norm(root_pos[:2] - self.goal_pos[:2])
        tracking_reward = (self.previous_tracking_distance - current_tracking_distance) * 20

        self.previous_tracking_distance = current_tracking_distance

        energy_reward = np.mean(self.curr_power_usage)
        self.curr_power_usage = []

        success_reward = 100 if current_tracking_distance < 0.1 else 0

        reward = (tracking_reward * (1 - self.reward_specs["w_energy"] - self.reward_specs["w_upright"]) + 
                  energy_reward * self.reward_specs["w_energy"] +
                  self.compute_upright_reward() * self.reward_specs["w_upright"] +
                  success_reward)

        self.reward_info = {
            "tracking_reward": tracking_reward,
            "upright_reward": self.compute_upright_reward(),
            "energy_reward": energy_reward,
            "success_reward": success_reward,
        }

        return reward

    def step(self, action):
        """
        Executes a single step in the environment with the given action.

        Args:
            action: The action to apply at the current step.

        Returns:
            Tuple:
                - observation (np.ndarray): Current observations after the step.
                - reward (float): Reward for the applied action.
                - terminated (bool): Whether the task has terminated prematurely.
                - truncated (bool): Whether the task has exceeded its allowed time.
                - info (dict): Additional information about the step, including reward details.
        """
        if self.recording_biomechanics:
            self.policy_outputs.append(action)

        self.physics_step(action)
        observation, reward, terminated, truncated, info = self.post_physics_step(action)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info
    
    def physics_step(self, action: np.ndarray = None) -> None:
        """
        Executes a physics step in the simulation with the given action.

        Depending on the control mode, computes muscle activations and applies them 
        to the simulation. Tracks power usage during the step.

        Args:
            action (np.ndarray): The action to apply. If None, a random action is sampled.
        """
        self.curr_power_usage = []

        if action is None:
            action = self.action_space.sample()
        
        if self.control_mode == "PD":
            target_lengths = action_to_target_length(action, self.mj_model)

        for i in range(self.control_freq_inv):
            if not self.paused:
                if self.control_mode == "PD":
                    muscle_activity = target_length_to_activation(target_lengths, self.mj_data, self.mj_model)
                elif self.control_mode == "direct":
                    muscle_activity = (action + 1.0) / 2.0

                else:
                    raise NotImplementedError
                  
                self.mj_data.ctrl[:] = muscle_activity
                mujoco.mj_step(self.mj_model, self.mj_data)
                self.curr_power_usage.append(self.compute_energy_reward(muscle_activity))