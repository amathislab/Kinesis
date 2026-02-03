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
# 2. MyoSuite (https://github.com/MyoHub/myosuite)
#    Copyright (c) MyoSuite Authors
#    Authors  :: Sherwin Chan (sherwin.chan@ntu.edu.sg), J-Anne Yow (janne.yow@ntu.edu.sg), Chun Kwang Tan (cktan.neumove@gmail.com), Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Pierre Schumacher (schumacherpier@gmail.com)
#    Licensed under the Apache License, Version 2.0

import numpy as np
from typing import Tuple
import mujoco
from omegaconf import DictConfig

from src.env.myolegs_im import MyoLegsIm, compute_imitation_observations
from src.env.myolegs_env import action_to_target_length, target_length_to_activation, MyoLegsEnv
from src.utils.visual_capsule import add_visual_capsule
from src.utils.quat_math import quat2euler, euler2quat
from src.utils.tracking_constants import *

from scipy.spatial.transform import Rotation as sRot

import logging

logger = logging.getLogger(__name__)

MYOLEG_TRACKED_BODIES = [
    "root",
    "head",
    "tibia_l",
    "tibia_r",
    "talus_l",
    "talus_r",
    "toes_l",
    "toes_r",
]

class MyoLegsBallReach(MyoLegsIm):
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
        self.global_offset = np.zeros([1, 3])
        self.gender_betas = [np.zeros(17)]  # current, all body shape is mean.

        self.initialize_tracking_constants(cfg)
        self.initialize_env_params(cfg)
        self.initialize_run_params(cfg)

        MyoLegsEnv.__init__(self, cfg)
        
        self.initialize_biomechanical_recording()
        self.initialize_evaluation_metrics()
        self.target_pos = np.array([40, 0, 0.94], dtype=np.float32)  # Initial ball position
        self.previous_tracking_distance = None

        if self.cfg.run.test == True:
            self.results_list = []

        self.myo_joints = ['Abs_r3', 'Abs_t1', 'Abs_t2', 'L1_L2_AR', 'L1_L2_FE', 'L1_L2_LB', 'L2_L3_AR', 'L2_L3_FE', 'L2_L3_LB', 
                            'L3_L4_AR', 'L3_L4_FE', 'L3_L4_LB', 'L4_L5_AR', 'L4_L5_FE', 'L4_L5_LB', 
                            'ankle_angle_l', 'ankle_angle_r', 'axial_rotation', 'flex_extension', 'hip_adduction_l', 
                            'hip_adduction_r', 'hip_flexion_l', 'hip_flexion_r', 'hip_rotation_l', 'hip_rotation_r', 
                            'knee_angle_l', 'knee_angle_l_beta_rotation1', 'knee_angle_l_beta_translation1', 'knee_angle_l_beta_translation2',
                             'knee_angle_l_rotation2', 'knee_angle_l_rotation3', 'knee_angle_l_translation1', 'knee_angle_l_translation2',
                             'knee_angle_r', 'knee_angle_r_beta_rotation1', 'knee_angle_r_beta_translation1', 'knee_angle_r_beta_translation2', 
                             'knee_angle_r_rotation2', 'knee_angle_r_rotation3', 'knee_angle_r_translation1', 'knee_angle_r_translation2', 
                             'lat_bending', 'mtp_angle_l', 'mtp_angle_r', 'subtalar_angle_l', 'subtalar_angle_r']
        
        self.ball_pos = self.get_ball_pos()

        self.goalkeeper = GoalKeeper(sim=self.mj_data,
                                     rng=self.np_random,
                                     probabilities=(0.0, 0.0, 1.0),
                                     random_vel_range=(5.0, 5.0))
        
        self.goalkeeper.dt = self.mj_model.opt.timestep * self.control_freq_inv

        self.success_flag = False

    def initialize_tracking_constants(self, cfg: DictConfig) -> None:
        """
        Initializes tracking constants for the environment.

        Args:
            cfg (DictConfig): Configuration object.
        Sets:
            - `tracked_bodies`: List of body parts to track.
            - `reset_bodies`: List of body parts to check for reset conditions.
            - `tracked_ids`: List of SMPL joint IDs to track.
            - `reset_ids`: List of SMPL joint IDs to check for reset conditions.
        """
        if cfg.project == "kinesis_legs":
            self.tracked_bodies = MYOLEG_TRACKED_BODIES
            self.reset_bodies = MYOLEG_RESET_BODIES
            self.smpl_tracked_ids = SMPL_TRACKED_IDS
            self.smpl_reset_ids = SMPL_RESET_IDS
        # elif cfg.project == "kinesis_fullbody":
        #     self.tracked_bodies = MYOLEG_FULLBODY_TRACKED_BODIES
        #     self.reset_bodies = MYOLEG_FULLBODY_RESET_BODIES
        #     self.smpl_tracked_ids = SMPL_FULLBODY_TRACKED_IDS
        #     self.smpl_reset_ids = SMPL_FULLBODY_RESET_IDS
        elif cfg.project == "kinesis_legs_abs" or cfg.project == "kinesis_legs_back":
            self.tracked_bodies = MYOLEG_ABS_TRACKED_BODIES
            self.reset_bodies = MYOLEG_ABS_RESET_BODIES
            self.smpl_tracked_ids = SMPL_ABS_TRACKED_IDS
            self.smpl_reset_ids = SMPL_ABS_RESET_IDS
        else:
            raise NotImplementedError(f"Project {cfg.project} not implemented.")

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
        if self.viewer is not None and self.target_pos is not None:
            self.viewer.user_scn.geoms[0].pos = self.target_pos

    def compute_tracking_distance(self) -> float:
        """
        Computes the Euclidean distance between the root position and the ball position.

        The distance is calculated in the 2D plane (ignoring the z-axis).

        Returns:
            float: The tracking distance.
        """
        body_pos = self.get_body_xpos()
        root_pos = body_pos[0]
        tracking_distance = np.linalg.norm(root_pos[:2] - self.target_pos[:2])
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
        self._set_reset_state(random=True)
        mujoco.mj_kinematics(self.mj_model, self.mj_data)


    def _set_reset_state(self, random=True, jnt_scale=0.01, pos_scale=0.5) -> Tuple[np.ndarray, np.ndarray]:
        self.mj_data.qpos = self.mj_model.key_qpos[0].copy()
        self.mj_data.qvel = self.mj_model.key_qvel[0].copy()

        if random:
            for jnt in self.myo_joints:
                self.mj_data.joint(jnt).qpos[0] += self.np_random.uniform(-np.abs(jnt_scale), 
                                                                           np.abs(jnt_scale), 
                                                                           size=(1,))
            self.mj_data.joint('root').qpos[0] += self.np_random.uniform(-np.abs(pos_scale), 
                                                                          0, 
                                                                          size=(1,)) # Only allow body to move behind the ball, not in front
            # self.mj_data.joint('root').qpos[1] += self.np_random.uniform(-np.abs(pos_scale) / 4, 
            #                                                               np.abs(pos_scale) / 4,
            #                                                               size=(1,))
            random_direction = self.np_random.choice([-1, 1])
            if random_direction == 1:
                self.mj_data.joint('root').qpos[1] += 0.3
                # self.mj_data.joint('root').qpos[1] += self.np_random.uniform(np.abs(pos_scale) / 6, 
                #                                                           np.abs(pos_scale) / 3,
                #                                                           size=(1,))
            else:
                self.mj_data.joint('root').qpos[1] -= 0.1
                # self.mj_data.joint('root').qpos[1] += self.np_random.uniform(-np.abs(pos_scale) / 6, 
                #                                                             -np.abs(pos_scale) / 8,
                #                                                             size=(1,)) # Y-direction movement allowable


    def compute_reset(self) -> Tuple[bool, bool]:
        """
        Determines whether the task should be reset based on termination and truncation conditions.

        Termination means failure, while truncation means success.

        Returns:
            tuple:
                - terminated (bool): If the agent falls before the episode times out, or if the episode times out but the agent is not close enough to the goal.
                - truncated (bool): If the agent is standing close enough to the goal when the episode times out.
        """
        # fall_terminated = self.proprioception["root_height"] < 0.7
        fall_terminated = self.head_height < 0.0
        timetout_terminated = self.cur_t >= self.cfg.env.max_episode_length
        truncated = timetout_terminated and np.linalg.norm(self.get_body_xpos()[0, :2] - self.target_pos[:2]) < self.cfg.env.success_radius
        terminated = fall_terminated or (timetout_terminated and not truncated)

        self.ball_pos = self.get_ball_pos()

        if self.ball_pos[0] > 50.0 and self.ball_pos[1] > -3.3 and self.ball_pos[1] < 3.3:
            self.success_flag = True

        if self.cfg.run.test == True:
            if truncated:
                self.results_list.append(True)
            elif terminated:
                self.results_list.append(False)

        return terminated or truncated, self.success_flag
    
    def reset_task(self, options=None) -> None:
        """
        Resets the task to an initial state by reinitializing key motion parameters.

        The function selects a random start time for the reference motion that the agent
        uses to initialize its position.

        Args:
            options (dict, optional): Additional options for task resetting. Defaults to None.
        """
        self.goalkeeper.reset_goalkeeper(rng=self.np_random)
        self.success_flag = False
    
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

        self.head_height = body_pos[0, 11, 2]

        root_rot = body_rot[:, 0]
        root_pos = body_pos[:, 0]

        body_pos_subset = body_pos[..., self.tracked_bodies_id, :]

        ref_pos_subset = body_pos_subset
        ref_pos_subset[..., 0, :] = self.target_pos

        body_vel = self.get_body_linear_vel()[None,]
        body_vel_subset = body_vel[..., self.tracked_bodies_id, :]

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

        current_tracking_distance = np.linalg.norm(root_pos[:2] - self.target_pos[:2])
        tracking_reward = (self.previous_tracking_distance - current_tracking_distance) * self.reward_specs["k_tracking"]

        self.previous_tracking_distance = current_tracking_distance

        energy_reward = np.mean(self.curr_power_usage)
        self.curr_power_usage = []

        position_reward = 1.0 if current_tracking_distance < self.cfg.env.success_radius else 0.0

        orientation_reward = self.compute_orientation_reward(current_tracking_distance)

        ball_pos = self.get_ball_pos()
        ball_vel = ball_pos - self.ball_pos

        ball_reward = self.compute_ball_reward(ball_vel)

        reward = (tracking_reward * self.reward_specs["w_tracking"] + 
                  energy_reward * self.reward_specs["w_energy"] +
                  self.compute_upright_reward() * self.reward_specs["w_upright"] +
                #   position_reward * self.reward_specs["w_position"] +
                #   orientation_reward * self.reward_specs["w_orientation"] +
                  ball_reward * self.reward_specs["w_ball"])

        self.reward_info = {
            "tracking_reward": tracking_reward,
            "upright_reward": self.compute_upright_reward(),
            "energy_reward": energy_reward,
            "position_reward": position_reward,
            "orientation_reward": orientation_reward,
            "ball_reward": ball_reward,
        }

        return reward

    def compute_orientation_reward(self, current_tracking_distance):
        if current_tracking_distance >= self.cfg.env.success_radius:
            return 0.0
        curr_angle = sRot.from_quat(self.get_qpos()[3:7])
        target_angle = sRot.from_quat(np.array([1, 0, 0, 0]))
        angle_diff = target_angle.inv() * curr_angle
        return np.exp(- self.reward_specs["k_orientation"] * angle_diff.magnitude())  # Encourage alignment with the target orientation
    
    def get_ball_pos(self):
        return self.mj_data.xpos[1].copy()
    
    def compute_ball_reward(self, ball_vel):
        if ball_vel[0] < 1e-2:
            return 0.0
        if ball_vel[0] / (np.abs(ball_vel[1]) + 1e-6) < 1.0:
            return 0.0
        return ball_vel[0] / self.dt  # Reward is the velocity of the ball in the x-direction

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

        self.ball_pos = self.get_ball_pos()

        self.goalkeeper.update_goalkeeper_state()

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

    def setup_myolegs_params(self) -> None:
        """
        Sets up various parameters related to the MyoLeg environment.
        """
        self.mj_body_names = []
        for i in range(self.mj_model.nbody):
            body_name = self.mj_model.body(i).name
            self.mj_body_names.append(body_name)
                
        self.body_names = self.mj_body_names[3:] # the first one is world, second is the ball
            
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

        self.full_tracked_bodies = self.body_names
        self.tracked_bodies_id = [
            self.body_names.index(j) for j in self.tracked_bodies
        ]
        self.reset_bodies_id = [
            self.body_names.index(j) for j in self.reset_bodies
        ]


class GoalKeeper:
    """
    GoalKeeper for the Soccer Track of the MyoChallenge 2025.
    Manages the goalkeeper's behaviour, including randomized movement speed
    and probabilities of blocking the goal.
    Contains several different policies. For the final evaluation, an additional
    non-disclosed policy will be used.
    """
    FIXED_X_POS = 50  # Initial position for the goalkeeper
    Y_MIN_BOUND = -3.3
    Y_MAX_BOUND = 3.3
    P_GAIN = 5.0
    
    def __init__(self,
                 sim,
                 rng,
                 probabilities: Tuple[float],
                 random_vel_range: Tuple[float],
                 dt=0.01,
        ):
        """
        Initialize the GoalKeeper class.
        :param sim: Mujoco sim object.
        :param rng: np_random generator.
        :param probabilities: Probabilities for the different policies, (stationary, random, block_ball).
        :param random_vel_range: Range of velocities for the block_ball policy. Clipped.
        :param dt: Simulation timestep.
        """
        self.dt = dt
        self.sim = sim
        self.goalkeeper_probabilities = probabilities
        self.random_vel_range = random_vel_range 
        self.reset_goalkeeper(rng=rng)

    def reset_noise_process(self):
        self.noise_process = BrownianNoiseProcess(size=(2, 2000), scale=10, rng=self.rng)

    def get_goalkeeper_pose(self):
        """
        Get goalkeeper Pose
        :return: The  pose.
        :rtype: list -> [x, y, angle]
        """
        angle = quat2euler(self.sim.mocap_quat[0, :])[-1]
        return np.concatenate([self.sim.mocap_pos[0, :2], [angle]])

    def set_goalkeeper_pose(self, pose: list):
        """
        Set goalkeeper pose directly.
        :param pose: Pose of the goalkeeper.
        :type pose: list -> [x, y, angle]
        """
        # Enforce goalkeeper limits
        pose[0] = self.FIXED_X_POS   
        pose[1] = np.clip(pose[1], self.Y_MIN_BOUND, self.Y_MAX_BOUND) 

        self.sim.mocap_pos[0, :2] = pose[:2]
        self.sim.mocap_quat[0, :] = euler2quat([0, 0, pose[-1]])

    def move_goalkeeper(self, vel: list):
        """
        This is the main function that moves the goalkeeper and should always be used if you want to physically move
        it by giving it a velocity. If you want to teleport it to a new position, use `set_goalkeeper_pose`.
        :param vel: Linear and rotational velocities in [-1, 1]. Moves goalkeeper
                  forwards or backwards and turns it. vel[0] is assumed to be linear vel and
                  vel[1] is assumed to be rotational vel
        :type vel: list -> [lin_vel, rot_vel].
        """
        self.goalkeeper_vel = vel
        assert len(vel) == 2

        lin_vel = vel[0]
        pose = self.get_goalkeeper_pose()
        pose[1] += self.dt * lin_vel
        
        self.set_goalkeeper_pose(pose)      # Enforce goalkeeper limits here
    
    def random_movement(self):
        """
        This moves the goalkeeper randomly in a correlated
        pattern.
        """
        return np.clip(self.noise_process.sample(), -self.block_velocity, self.block_velocity)
    
    def block_ball_policy(self):
        """
        Calculates the linear velocity along the Y-axis required for the goalkeeper to
        move towards and block the ball.
        The goalkeeper's orientation and X-position are fixed.
        The target Y for the goalkeeper is clipped to its valid movement range.
        """
        goalkeeper_pose = self.get_goalkeeper_pose()
        ball_pos = self.sim.body('soccer_ball').xpos[:3].copy()

        # Clip the target Y position to the goalkeeper's allowed range
        target_y = np.clip(ball_pos[1], self.Y_MIN_BOUND, self.Y_MAX_BOUND)
        current_y = goalkeeper_pose[1]

        displacement = target_y - current_y

        linear_vel_y = np.clip(displacement, -self.block_velocity, self.block_velocity)
        
        return np.array([linear_vel_y * self.P_GAIN, 0.0])

    def sample_goalkeeper_policy(self):
        """
        Takes in three probabilities and returns the policies with the given frequency.
        """
        rand_num = self.rng.uniform()
        if rand_num < self.goalkeeper_probabilities[0]:
            self.goalkeeper_policy = 'stationary'
        elif rand_num < self.goalkeeper_probabilities[0] + self.goalkeeper_probabilities[1]:
            self.goalkeeper_policy = 'random'
        elif rand_num < self.goalkeeper_probabilities[0] + self.goalkeeper_probabilities[1] + self.goalkeeper_probabilities[2]:
            self.goalkeeper_policy = 'block_ball'

    def update_goalkeeper_state(self):
        """
        This function executes an goalkeeper step with
        one of the control policies.
        """
        if self.goalkeeper_policy == 'stationary':
            goalkeeper_vel = np.zeros(2,)

        elif self.goalkeeper_policy == 'random':
            goalkeeper_vel = self.random_movement()

        elif self.goalkeeper_policy == 'block_ball':
            goalkeeper_vel = self.block_ball_policy()
        else:
            raise NotImplementedError(f"This goalkeeper policy doesn't exist. Chose: stationary, random or block_ball. Policy was: {self.goalkeeper_policy}")
        self.move_goalkeeper(goalkeeper_vel)

    def reset_goalkeeper(self, rng=None):
        """
        Resets the goalkeeper's position, policy, and blocking speed.
        :rng: np_random generator
        """
        if rng is not None:
            self.rng = rng
            self.reset_noise_process()

        self.goalkeeper_vel = np.zeros((2,))
        self.sample_goalkeeper_policy()

        initial_goalkeeper_pos = [self.FIXED_X_POS, 0, 0]
        self.set_goalkeeper_pose(initial_goalkeeper_pos)
        self.goalkeeper_vel[:] = 0.0

        # Randomize the maximum linear speed for the 'block_ball' policy for this reset.
        # This value is used within the `block_ball_policy` to clip movement speed.
        self.block_velocity = self.rng.uniform(self.random_vel_range[0], self.random_vel_range[1])

class BrownianNoiseProcess:
    def __init__(self, size, scale=1.0, rng=None):
        self.size = size
        self.scale = scale
        self.rng = rng if rng is not None else np.random
        self.reset()

    def reset(self):
        self.white_noise = self.rng.normal(0, 1, self.size)
        self.brownian_noise = np.cumsum(self.white_noise, axis=1) * self.scale
        self.idx = 0

    def sample(self):
        if self.idx >= self.size[1]:
            self.reset()
        sample = self.brownian_noise[:, self.idx]
        self.idx += 1
        return sample