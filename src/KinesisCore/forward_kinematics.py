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

import torch
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from src.smpl.smpl_parser import (
    SMPL_Parser,
    SMPL_BONE_ORDER_NAMES,
)
from easydict import EasyDict
import scipy.ndimage as ndimage
import src.utils.pytorch3d_transforms as tRot

class ForwardKinematics:
    def __init__(self, data_dir):
        self.smpl_parser = SMPL_Parser(model_path=data_dir, gender="neutral")

        self.model_names = [
            "Pelvis",
            "L_Hip",
            "L_Knee",
            "L_Ankle",
            "L_Toe",
            "R_Hip",
            "R_Knee",
            "R_Ankle",
            "R_Toe",
            "Torso",
            "Spine",
            "Chest",
            "Neck",
            "Head",
            "L_Thorax",
            "L_Shoulder",
            "L_Elbow",
            "L_Wrist",
            "L_Hand",
            "R_Thorax",
            "R_Shoulder",
            "R_Elbow",
            "R_Wrist",
            "R_Hand",
        ]
        self._parents = [
            -1,
            0,
            1,
            2,
            3,
            0,
            5,
            6,
            7,
            0,
            9,
            10,
            11,
            12,
            11,
            14,
            15,
            16,
            17,
            11,
            19,
            20,
            21,
            22,
        ]  # mujoco order SMPL parents.
        self.smpl_2_mujoco = [
            SMPL_BONE_ORDER_NAMES.index(i) for i in self.model_names
        ]  # Apply Mujoco order
        self.mujoco_2_smpl = [
            self.model_names.index(i) for i in SMPL_BONE_ORDER_NAMES
        ]  # Apply Mujoco order
        self.num_joints = len(self._parents)
        self.dt = 1 / 30
        self.update_model(
            torch.zeros((1, 10))
        )  # default gender 0 and pose 0.

    def update_model(self, betas, dt = 1/30):
        betas = betas.cpu().float()
        (
                _,
                _,
                _,
                _,
                joint_offsets,
                _,
                _,
                _,
                _,
                _,
                _,
        ) = self.smpl_parser.get_mesh_offsets_batch(betas=betas[:, :10])

        reordered_joint_offsets = dict()
        for n in SMPL_BONE_ORDER_NAMES:
            reordered_joint_offsets[n] = joint_offsets[n]
        
        offsets = []
        for n in self.model_names:
            offsets.append(reordered_joint_offsets[n])

        self._offsets = torch.from_numpy(
            np.round(np.stack(offsets, axis=1), decimals=5)
        )

        self.dt = dt

    def fk_batch(
            self,
            pose_aa: torch.Tensor,
            trans: torch.Tensor,
    ):
        """
        Performs batched forward kinematics (FK) for SMPL pose and translation inputs, producing Mujoco-compatible outputs.

        Args:
            pose (torch.Tensor): Input poses in axis-angle format of shape `(B, T, J, 3)`, where:
                - `B`: Batch size.
                - `T`: Sequence length.
                - `J`: Number of joints.
            trans (torch.Tensor): Global translations of shape `(B, T, 3)`.
            
        Returns:
            EasyDict
        """
        device, dtype = pose_aa.device, pose_aa.dtype
        assert len(pose_aa.shape) == 4
        B, T = pose_aa.shape[:2]
        pose_quat = tRot.axis_angle_to_quaternion(pose_aa)
        pose_mat = tRot.quaternion_to_matrix(pose_quat)

        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(B, T, -1, 3, 3)

        trans = trans + self._offsets[:, 0:1].to(device)

        pose_mat_ordered = pose_mat[:, :, self.smpl_2_mujoco]

        wbody_pos, wbody_mat = self.forward_kinematics_batch(
            pose_mat_ordered[:, :, 1:], pose_mat_ordered[:, :, 0:1], trans
        )

        return_dict = EasyDict()
        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat

        
        wbody_rot = tRot.matrix_to_quaternion(wbody_mat)
        rigidbody_linear_velocity = self._compute_velocity(
            wbody_pos, self.dt,
        )
        rigidbody_angular_velocity = self._compute_angular_velocity(
            wbody_rot, self.dt
        )
        return_dict.global_rotation = wbody_rot
        return_dict.local_rotation = pose_quat
        return_dict.global_root_velocity = rigidbody_linear_velocity[..., 0, :]
        return_dict.global_root_angular_velocity = rigidbody_angular_velocity[
            ..., 0, :
        ]

        return_dict.global_angular_velocity = rigidbody_angular_velocity
        return_dict.global_velocity = rigidbody_linear_velocity

        dof_pos = tRot.matrix_to_euler_angles(pose_mat_ordered, "XYZ")[..., 1:, :]

        return_dict.dof_pos = torch.cat(
            [tRot.fix_continous_dof(dof_pos_t)[None,] for dof_pos_t in dof_pos],
            dim=0,
        )

        dof_vel = (
            return_dict.dof_pos[:, 1:] - return_dict.dof_pos[:, :-1]
        ) / self.dt
        while len(dof_vel[dof_vel > np.pi]) > 0:
            dof_vel[dof_vel > np.pi] -= 2 * np.pi
        while len(dof_vel[dof_vel < -np.pi]) > 0:
            dof_vel[dof_vel < -np.pi] += 2 * np.pi
        return_dict.dof_vels = torch.cat([dof_vel, dof_vel[:, -1:]], dim=1)
        return_dict.fps = int(1 / self.dt)

        return_dict.qpos = torch.cat(
            [trans, pose_quat[..., 0, :], return_dict.dof_pos.view(B, T, -1)],
            dim=-1,
        )

        local_root_angular_velocity = (
            wbody_mat[
                :,
                :,
                0,
            ].transpose(3, 2)
            @ return_dict.global_root_angular_velocity[..., None]
        )[..., 0]
        return_dict.qvel = torch.cat(
            [
                return_dict.global_root_velocity,
                local_root_angular_velocity,
                return_dict.dof_vels.view(B, T, -1),
            ],
            dim=-1,
        )

        return return_dict

    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
            -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
            -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """

        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []
        expanded_offsets = (
            self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype)
        )

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (
                    torch.matmul(
                        rotations_world[self._parents[i]][:, :, 0],
                        expanded_offsets[:, :, i, :, None],
                    ).squeeze(-1)
                    + positions_world[self._parents[i]]
                )

                rot_mat = torch.matmul(
                    rotations_world[self._parents[i]], rotations[:, :, (i - 1) : i, :]
                )

                positions_world.append(jpos)
                rotations_world.append(rot_mat)

        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world

    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        assert len(p.shape) == 4

        velocity = (p[:, 1:, ...] - p[:, :-1, ...]) / time_delta
        velocity = torch.cat([velocity[:, :1, ...], velocity], dim=1)  # Mujoco

        if guassian_filter:
            velocity = torch.from_numpy(
                ndimage.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")
            ).to(p)

        return velocity

    @staticmethod
    def _compute_angular_velocity(rotations, time_delta: float, guassian_filter=True):
        # assume the second last dimension is the time axis

        diff_quat_data = tRot.quat_identity_like(rotations).to(rotations)

        diff_quat_data[..., 1:, :, :] = tRot.quat_mul_norm(
            rotations[..., 1:, :, :], tRot.quat_inverse(rotations[..., :-1, :, :])
        )
        diff_quat_data[..., 0, :, :] = diff_quat_data[..., 1, :, :]
        diff_angle, diff_axis = tRot.quat_angle_axis(diff_quat_data)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta

        if guassian_filter:
            angular_velocity = torch.from_numpy(
                ndimage.gaussian_filter1d(
                    angular_velocity.numpy(), 2, axis=-3, mode="nearest"
                ),
            )
        return angular_velocity