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

import os
import sys
from typing import List, Tuple
sys.path.append(os.getcwd())

import numpy as np
from collections import OrderedDict
import gymnasium as gym
import mujoco
from scipy.spatial.transform import Rotation as sRot

from src.env.myolegs_base_env import BaseEnv
import src.utils.np_transform_utils as npt_utils


class MyoLegsEnv(BaseEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(cfg=self.cfg)
        self.setup_configs(cfg)

        self.create_sim(
            cfg.run.xml_path
        )
        self.setup_myolegs_params()
        self.reward_info = {}
        
        self.observation_space = gym.spaces.Box(
            -np.inf * np.ones(self.get_obs_size()),
            np.inf * np.ones(self.get_obs_size()),
            dtype=self.dtype,
        )
        
        self.action_space = gym.spaces.Box(
            low=-np.ones(80),
            high=np.ones(80),
            dtype=self.dtype,
        )

    def setup_configs(self, cfg) -> None:
        """
        Sets various configuration parameters.
        """
        self._kp_scale = cfg.env.kp_scale
        self._kd_scale = cfg.env.kd_scale
        self.control_mode = cfg.run.control_mode
        self.max_episode_length = 300
        self.dtype = np.float32

    def setup_myolegs_params(self) -> None:
        """
        Sets up various parameters related to the MyoLeg environment.
        """
        self.mj_body_names = []
        for i in range(self.mj_model.nbody):
            body_name = self.mj_model.body(i).name
            self.mj_body_names.append(body_name)
        
        self.body_names = self.mj_body_names[1:] # the first one is always world
            
        self.num_bodies = len(self.body_names)
        self.num_vel_limit = self.num_bodies * 3
        self.robot_body_idxes = [
            self.mj_body_names.index(name) for name in self.body_names
        ]
        self.robot_idx_start = self.robot_body_idxes[0]
        self.robot_idx_end = self.robot_body_idxes[-1] + 1

        self.qpos_lim = np.max(self.mj_model.jnt_qposadr) + self.mj_model.jnt_qposadr[-1] - self.mj_model.jnt_qposadr[-2]
        self.qvel_lim = np.max(self.mj_model.jnt_dofadr) + self.mj_model.jnt_dofadr[-1] - self.mj_model.jnt_dofadr[-2]
        
        # These are not required but are included for future reference
        geom_type_id = mujoco.mju_str2Type("geom")
        self.floor_idx = mujoco.mj_name2id(self.mj_model, geom_type_id, "floor")
    
    def get_obs_size(self) -> int:
        """
        Returns the size of the observations. In the environment class, this defaults to the size of the proprioceptive observations.
        """
        return self.get_self_obs_size()

    def compute_observations(self) -> np.ndarray:
        """
        Computes the observations. In the environment class, this defaults to the proprioceptive observations.
        """
        obs = self.compute_proprioception()
        return obs
    
    def compute_info(self):
        raise NotImplementedError
    
    def get_self_obs_size(self) -> int:
        """
        Returns the size of the proprioceptive observations.
        """
        inputs = self.cfg.run.proprioceptive_inputs
        tally = 0
        if "root_height" in inputs:
            tally += 1
        if "root_tilt" in inputs:
            tally += 4
        if "local_body_pos" in inputs:
            tally += 3 * self.num_bodies - 3
        if "local_body_rot" in inputs:
            tally += 6 * self.num_bodies
        if "local_body_vel" in inputs:
            tally += 3 * self.num_bodies
        if "local_body_ang_vel" in inputs:
            tally += 3 * self.num_bodies
        if "muscle_len" in inputs:
            tally += self.mj_model.nu
        if "muscle_vel" in inputs:
            tally += self.mj_model.nu
        if "muscle_force" in inputs:
            tally += self.mj_model.nu
        if "feet_contacts" in inputs:
            tally += 4

        return tally

    def compute_proprioception(self) -> np.ndarray:
        """
        Computes proprioceptive observations for the current simulation state.

        Updates the humanoid's body and actuator states, and generates observations 
        based on the configured inputs.

        Returns:
            np.ndarray: Flattened array of proprioceptive observations.

        Notes:
            - The observations are also stored in the `self.proprioception` attribute.
        """
        mujoco.mj_kinematics(self.mj_model, self.mj_data)  # update xpos to the latest simulation values
        
        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]
        
        body_vel = self.get_body_linear_vel()[None,]
        body_ang_vel = self.get_body_angular_vel()[None,]

        obs_dict =  compute_self_observations(body_pos, body_rot, body_vel, body_ang_vel)
        
        root_rot = sRot.from_quat(self.mj_data.qpos[[4, 5, 6, 3]])
        root_rot_euler = root_rot.as_euler("xyz")

        myolegs_obs = OrderedDict()
        
        inputs = self.cfg.run.proprioceptive_inputs

        if "root_height" in inputs:
            myolegs_obs["root_height"] = obs_dict["root_h_obs"] # 1
        if "root_tilt" in inputs:
            myolegs_obs["root_tilt"] = np.array([np.cos(root_rot_euler[0]), np.sin(root_rot_euler[0]), np.cos(root_rot_euler[1]), np.sin(root_rot_euler[1])]) # 4
        if "local_body_pos" in inputs:
            myolegs_obs["local_body_pos"] = obs_dict["local_body_pos"][0] # 3 * num_bodies
        if "local_body_rot" in inputs:
            myolegs_obs["local_body_rot"] = obs_dict["local_body_rot_obs"][0] # 6 * num_bodies
        if "local_body_vel" in inputs:
            myolegs_obs["local_body_vel"] = obs_dict["local_body_vel"][0] # 3 * num_bodies
        if "local_body_ang_vel" in inputs:
            myolegs_obs["local_body_ang_vel"] = obs_dict["local_body_ang_vel"][0] # 3 * num_bodies
        if "muscle_len" in inputs:
            myolegs_obs["muscle_len"] = np.nan_to_num(self.mj_data.actuator_length.copy()) # num_actuators
        if "muscle_vel" in inputs:
            myolegs_obs["muscle_vel"] = np.nan_to_num(self.mj_data.actuator_velocity.copy()) # num_actuators
        if "muscle_force" in inputs:
            myolegs_obs["muscle_force"] = np.nan_to_num(self.mj_data.actuator_force.copy()) # num_actuators
        if "feet_contacts" in inputs:
            myolegs_obs["feet_contacts"] = self.get_touch() # 4
        self.proprioception = myolegs_obs

        return np.concatenate([v.ravel() for v in myolegs_obs.values()], axis=0, dtype=self.dtype)
    
    def get_body_xpos(self):
        """
        Returns the body positions of the agent in X, Y, Z coordinates.
        """
        return self.mj_data.xpos.copy()[self.robot_idx_start : self.robot_idx_end]

    def get_body_xquat(self):
        """
        Returns the body rotations of the agent in quaternion
        """
        return self.mj_data.xquat.copy()[self.robot_idx_start : self.robot_idx_end]
    
    def get_body_linear_vel(self):
        """
        Returns the linear velocity of the agent's body parts.
        """
        return self.mj_data.sensordata[:self.num_vel_limit].reshape(self.num_bodies, 3).copy()
    
    def get_body_angular_vel(self):
        """
        Returns the angular velocity of the agent's body parts.
        """
        return self.mj_data.sensordata[self.num_vel_limit:2 * self.num_vel_limit].reshape(self.num_bodies, 3).copy()
    
    def get_touch(self):
        """
        Returns the touch sensor readings of the agent.
        """
        return self.mj_data.sensordata[self.num_vel_limit * 2:].copy()
        
    def get_qpos(self):
        """
        Returns the joint positions of the agent.
        """
        return self.mj_data.qpos.copy()[: self.qpos_lim]

    def get_qvel(self):
        """
        Returns the joint velocities of the agent.
        """
        return self.mj_data.qvel.copy()[:self.qvel_lim]
    
    def get_root_pos(self):
        """
        Returns the position of the agent's root.
        """
        return self.get_body_xpos()[0].copy()

    def compute_reward(self, action):
        """
        Placeholder for reward computation. In the environment class, this defaults to 0.
        """
        reward = 0
        return reward

    def compute_reset(self) -> Tuple[bool, bool]:
        """
        Determines whether the episode should reset based on termination and truncation conditions.

        In the environment class, the episode ends if the current time step exceeds the maximum episode length.
        """
        if self.cur_t > self.max_episode_length:
            return False, True
        else:
            return False, False

    def pre_physics_step(self, action):
        """
        Placeholder for pre-physics-step computations. In the environment class, this defaults to no operation
        """
        pass

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

                    # MANUAL MUSCLE DEACTIVATION
                    if self.cfg.run.deactivate_muscles:
                        inactive_muscles = ["tibant_l", "tibant_r"]
                        muscle_activity = self.deactivate_muscles(muscle_activity, inactive_muscles)

                elif self.control_mode == "direct":
                    muscle_activity = (action + 1.0) / 2.0

                else:
                    raise NotImplementedError
                  
                self.mj_data.ctrl[:] = muscle_activity
                mujoco.mj_step(self.mj_model, self.mj_data)
                self.curr_power_usage.append(self.compute_energy_reward(muscle_activity))
    
    def deactivate_muscles(self, muscle_activity: np.ndarray, targetted_muscles: List[str]) -> np.ndarray:
        """
        Deactivates specific muscles by setting their activation values to zero.

        Args:
            muscle_activity (np.ndarray): Array of muscle activation values.
            targetted_muscles (list): List of muscle names (str) to deactivate.

        Returns:
            np.ndarray: Updated muscle activation values with the targeted muscles deactivated.
        """
        muscle_names = get_actuator_names(self.mj_model)
        indexes = [muscle_names.index(muscle) for muscle in targetted_muscles]
        for idx in indexes:
            muscle_activity[idx] = 0.0
        return muscle_activity

    def post_physics_step(self, action):
        """
        Processes the environment state after the physics step.

        Increments the simulation time, computes observations, reward, and checks 
        for termination or truncation conditions. Collects and returns additional 
        information about the reward components.

        Args:
            action (np.ndarray): The action applied at the current step.

        Returns:
            Tuple:
                - obs (np.ndarray): Current observations.
                - reward (float): Reward for the current step.
                - terminated (bool): Whether the task has terminated prematurely.
                - truncated (bool): Whether the task has exceeded its allowed time.
                - info (dict): Additional information, including raw reward components.
        """
        if not self.paused:
            self.cur_t += 1
        obs = self.compute_observations()
        reward = self.compute_reward(action)
        terminated, truncated = self.compute_reset()
        if self.disable_reset:
            terminated, truncated = False, False
        info = {}
        info.update(self.reward_info)
        return obs, reward, terminated, truncated, info
    
    def init_myolegs(self):
        """
        Initializes the MyoLegs environment. In the environment class, this defaults to
        setting the agent to a default position.
        """
        self.mj_data.qpos[:] = 0
        self.mj_data.qvel[:] = 0
        self.mj_data.qpos[2] = 0.94
        self.mj_data.qpos[3:7] = np.array([0.5, 0.5, 0.5, 0.5])   

    def reset_myolegs(self):
        self.init_myolegs()
        
    def forward_sim(self):
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def reset(self, seed=None, options=None):
        self.reset_myolegs()
        self.forward_sim()
        return super().reset(seed=seed, options=options)


def compute_self_observations(body_pos: np.ndarray, body_rot: np.ndarray, body_vel: np.ndarray, body_ang_vel: np.ndarray) -> OrderedDict:
    """
    Computes observations of the agent's local body state relative to its root.

    Args:
        body_pos (np.ndarray): Global positions of the bodies.
        body_rot (np.ndarray): Global rotations of the bodies in quaternion format.
        body_vel (np.ndarray): Linear velocities of the bodies.
        body_ang_vel (np.ndarray): Angular velocities of the bodies.

    Returns:
        OrderedDict: Dictionary containing:
            - `root_h_obs`: Root height observation.
            - `local_body_pos`: Local body positions excluding root.
            - `local_body_rot_obs`: Local body rotations in tangent-normalized format.
            - `local_body_vel`: Local body velocities.
            - `local_body_ang_vel`: Local body angular velocities.
    """
    obs = OrderedDict()
    
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]
    
    heading_rot_inv = npt_utils.calc_heading_quat_inv(root_rot)
    root_h = root_pos[:, 2:3]

    obs["root_h_obs"] = root_h
    
    heading_rot_inv_expand = heading_rot_inv[..., None, :]
    heading_rot_inv_expand = heading_rot_inv_expand.repeat(body_pos.shape[1], axis=1)
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],heading_rot_inv_expand.shape[2],)

    root_pos_expand = root_pos[..., None, :]
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = npt_utils.quat_rotate(
        flat_heading_rot_inv, flat_local_body_pos
    )
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    obs["local_body_pos"] = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )  # This is global rotation of the body
    flat_local_body_rot = npt_utils.quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = npt_utils.quat_to_tan_norm(flat_local_body_rot)
    obs["local_body_rot_obs"] = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    ###### Velocity ######
    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = npt_utils.quat_rotate(flat_heading_rot_inv, flat_body_vel)
    obs["local_body_vel"]  = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = npt_utils.quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    obs["local_body_ang_vel"] = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
    return obs

def get_actuator_names(model) -> list:
    """
    Retrieves the names of all actuators in the Mujoco model.

    Args:
        model: The Mujoco model containing actuator information.

    Returns:
        list: A list of actuator names as strings.
    """
    actuators = []
    for i in range(model.nu):
        if i == model.nu - 1:
            end_p = None
            for el in ["name_numericadr", "name_textadr", "name_tupleadr", "name_keyadr", "name_pluginadr", "name_sensoradr"]:
                v = getattr(model, el)
                if np.any(v):
                    if end_p is None:
                        end_p = v[0]
                    else:
                        end_p = min(end_p, v[0])
            if end_p is None:
                end_p = model.nnames
        else:
            end_p = model.name_actuatoradr[i+1]
        name = model.names[model.name_actuatoradr[i]:end_p].decode("utf-8").rstrip('\x00')
        actuators.append(name)
    return actuators

def force_to_activation(forces, model, data):
    """
    Converts actuator forces to activation levels for each actuator in the Mujoco model.

    Args:
        forces (np.ndarray): Array of forces applied to the actuators.
        model: The Mujoco model containing actuator properties.
        data: The Mujoco data structure with runtime actuator states.

    Returns:
        list: Activation levels for each actuator, clipped between 0 and 1.
    """
    activations = []
    for idx_actuator in range(model.nu):
        length = data.actuator_length[idx_actuator]
        lengthrange = model.actuator_lengthrange[idx_actuator]
        velocity = data.actuator_velocity[idx_actuator]
        acc0 = model.actuator_acc0[idx_actuator]
        prmb = model.actuator_biasprm[idx_actuator,:9]
        prmg = model.actuator_gainprm[idx_actuator,:9]
        bias = mujoco.mju_muscleBias(length, lengthrange, acc0, prmb)
        gain = min(-1, mujoco.mju_muscleGain(length, velocity, lengthrange, acc0, prmg))
        activations.append(np.clip((forces[idx_actuator] - bias) / gain, 0, 1))

    return activations

def target_length_to_force(lengths: np.ndarray, data, model) -> list:
    """
    Converts target muscle lengths to forces using a PD control law.

    Args:
        lengths (np.ndarray): Target lengths for the actuators.
        data: Mujoco data structure containing current actuator states.
        model: Mujoco model containing actuator properties.

    Returns:
        list: Clipped forces for each actuator, constrained by peak force.
    """
    forces = []
    for idx_actuator in range(model.nu):
        length = data.actuator_length[idx_actuator]
        velocity = data.actuator_velocity[idx_actuator]
        peak_force = model.actuator_biasprm[idx_actuator, 2]
        kp = 5 * peak_force
        kd = 0.1 * kp
        force = (kp * (lengths[idx_actuator] - length) - kd * velocity)
        clipped_force = np.clip(force, -peak_force, 0)
        forces.append(clipped_force)

    return forces

def target_length_to_activation(lengths: np.ndarray, data, model) -> np.ndarray:
    """
    Converts target lengths to activation levels via force computation.

    Args:
        lengths (np.ndarray): Target lengths for the actuators.
        data: Mujoco data structure containing current actuator states.
        model: Mujoco model containing actuator properties.

    Returns:
        np.ndarray: Activation levels for each actuator, clipped between 0 and 1.
    """
    forces = target_length_to_force(lengths, data, model)
    activations = force_to_activation(forces, model, data)
    return np.clip(activations, 0, 1)

def action_to_target_length(action: np.ndarray, model) -> list:
    """
    Maps actions to target lengths for actuators based on their length ranges.

    Args:
        action (np.ndarray): Action values in the range [-1, 1].
        model: Mujoco model containing actuator length range properties.

    Returns:
        list: Target lengths for each actuator.
    """
    target_lengths = []
    for idx_actuator in range(model.nu):
        # Set high to max length and low=0
        hi = model.actuator_lengthrange[idx_actuator, 1]
        lo = 0
        target_lengths.append((action[idx_actuator] + 1) / 2 * (hi - lo) + lo)
    return target_lengths