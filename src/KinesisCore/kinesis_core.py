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
from typing import List, Tuple, Union

sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm

import torch
import torch.multiprocessing as mp

import joblib

from scipy.spatial.transform import Rotation as sRot
import random
random.seed(0)
from src.utils.torch_utils import to_torch
from easydict import EasyDict

from src.KinesisCore.forward_kinematics import ForwardKinematics

torch.set_num_threads(1)

class KinesisCore:

    def __init__(self, config):
        self.config = config
        self.dtype = np.float32

        self.load_data(config.motion_file)
        self._curr_motion_ids = None
        self._sampling_prob = (
            np.ones(self._num_unique_motions) / self._num_unique_motions
        )
        # Ignoring mesh parsers for now
        self.mesh_parsers = None

        self.fk_model = ForwardKinematics(config.data_dir)
        # self.fk_model = Humanoid_Batch("smpl")

    def load_data(self, filepath: str) -> None:
        """
        Loads motion data from a given pickle file.

        Args:
            filepath: The path to the file containing the motion data (in pkl format).

        Returns:
            None

        Updates:
            self.motion_data: A dictionary containing the motion data.
            self._num_unique_motions: The number of unique motions in the dataset.

        Notes:
            The file should contain a dictionary, where the keys are strings
            representing the name of the motion, and each value is a dictionary
            with the following keys: 'pose_quat_global', 'pose_quat', 'trans_orig', 
            'root_trans_offset', 'beta', 'gender', 'pose_aa', 'fps'
        """
        self.motion_data = joblib.load(filepath)
        self._num_unique_motions = len(self.motion_data.keys())

    def load_motions(
            self,
            m_cfg: dict,
            shape_params: List[np.ndarray],
            random_sample: bool = True,
            start_idx: int = 0,
            silent: bool = False,
            specific_idxes: np.ndarray = None,
    ):
        
        motions = []
        motion_lengths = []
        motion_fps_acc = []
        motion_dt = []
        motion_num_frames = []
        motion_bodies = []
        motion_aa = []

        self.num_joints = 24

        num_motion_to_load = len(shape_params)  # This assumes that the number of shape params defines the number of motions to load
        if specific_idxes is not None:
            if len(specific_idxes) < num_motion_to_load:
                num_motion_to_load = len(specific_idxes)
            if random_sample:
                sample_idxes = np.random.choice(
                    specific_idxes,
                    size=num_motion_to_load,
                    replace=False,
                )
            else:
                sample_idxes = specific_idxes
        else:
            if random_sample:
                sample_idxes = np.random.choice(
                    np.arange(self._num_unique_motions),
                    size=num_motion_to_load,
                    replace=False,
                )
            else:
                sample_idxes = np.remainder(
                    np.arange(start_idx, start_idx + num_motion_to_load),
                    self._num_unique_motions,
                )

        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = [list(self.motion_data.keys())[i] for i in sample_idxes]
        
        self._sampling_batch_prob = np.ones(len(self._curr_motion_ids)) / len(
            self._curr_motion_ids
        )
        
        motion_data_list = [self.motion_data[self.curr_motion_keys[i]] for i in range(num_motion_to_load)]

        if sys.platform == "darwin":
            num_jobs = 1
        else:
            mp.set_sharing_strategy("file_descriptor")

        manager = mp.Manager()
        queue = manager.Queue()
        num_jobs = min(min(mp.cpu_count(), 64), num_motion_to_load)

        if len(motion_data_list) <= 32 or not self.config.multi_thread or num_jobs <= 8:
            num_jobs = 1

        res_acc = {}

        chunk = np.ceil(len(motion_data_list) / num_jobs).astype(int)
        ids = np.arange(len(motion_data_list))

        jobs = [
            (
                ids[i: i + chunk],
                motion_data_list[i: i + chunk],
                self.config,
            )
            for i in range(0, len(motion_data_list), chunk)
        ]
        for i in range(1, len(jobs)):
            worker_args = (*jobs[i], queue, i)
            worker = mp.Process(target=self.load_motions_worker, args=worker_args)
            worker.start()
        res_acc.update(self.load_motions_worker(*jobs[0], None, 0))
        pbar = tqdm(range(len(jobs) - 1))
        for i in pbar:
            res = queue.get()
            res_acc.update(res)
        pbar = tqdm(range(len(res_acc)))

        for f in pbar:
            curr_motion = res_acc[f]
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_translation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)
            motion_aa.append(curr_motion.pose_aa)
            motion_fps_acc.append(motion_fps)
            motion_dt.append(curr_dt)
            motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            motion_lengths.append(curr_len)

            del curr_motion

        self._motion_lengths = np.array(motion_lengths).astype(self.dtype)
        self._motion_fps = np.array(motion_fps_acc).astype(self.dtype)
        self._motion_aa = np.concatenate(motion_aa, axis=0).astype(self.dtype)
        self._motion_dt = np.array(motion_dt).astype(self.dtype)
        self._motion_num_frames = np.array(motion_num_frames)
        self._num_motions = len(motions)

        self.gts = np.concatenate(
            [m.global_translation for m in motions], axis=0
        ).astype(self.dtype)
        self.grs = np.concatenate(
            [m.global_rotation for m in motions], axis=0
        ).astype(self.dtype)
        self.lrs = np.concatenate(
            [m.local_rotation for m in motions], axis=0
        ).astype(self.dtype)
        self.grvs = np.concatenate(
            [m.global_root_velocity for m in motions], axis=0
        ).astype(self.dtype)
        self.gravs = np.concatenate(
            [m.global_root_angular_velocity for m in motions], axis=0
        ).astype(self.dtype)
        self.gavs = np.concatenate(
            [m.global_angular_velocity for m in motions], axis=0
        ).astype(self.dtype)
        self.gvs = np.concatenate([m.global_velocity for m in motions], axis=0).astype(
            self.dtype
        )
        self.dvs = np.concatenate([m.dof_vels for m in motions], axis=0).astype(
            self.dtype
        )
        self.dof_pos = np.concatenate([m.dof_pos for m in motions], axis=0).astype(
            self.dtype
        )
        self.qpos = np.concatenate([m.qpos for m in motions], axis=0).astype(self.dtype)
        self.qvel = np.concatenate([m.qvel for m in motions], axis=0).astype(self.dtype)

        lengths = self._motion_num_frames
        lengths_shifted = np.roll(lengths, 1, axis=0)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = np.arange(len(motions))
        self.num_bodies = self.num_joints

        num_motions = self._num_motions
        total_len = sum(self._motion_lengths)
        print(
            f"###### Sampling {num_motions:d} motions:",
            sample_idxes[:5],
            self.curr_motion_keys[:5],
            f"total length of {total_len:.3f}s and {self.gts.shape[0]} frames.",
        )

        return motions

    def load_motions_worker(
            self,
            ids: np.ndarray,
            motion_data_list: List[dict],
            config: dict,
            queue: Union[mp.Queue, None],
            pid: int,
    ):
        np.random.seed(np.random.randint(5000) * pid)
        res = {}
        for f in range(len(motion_data_list)):
            curr_id = ids[f]
            motion_data = motion_data_list[f]
            fps = motion_data.get("fps", 30)
            motion_length = motion_data["pose_aa"].shape[0]

            trans = (
                to_torch(
                    motion_data["trans"]
                    if "trans" in motion_data
                    else motion_data["trans_orig"]
                ).float().clone()
            )

            pose_aa = to_torch(motion_data["pose_aa"]).float().clone()
            if pose_aa.shape[1] == 156:
                pose_aa = torch.cat(
                    [pose_aa[:, :66], torch.zeros((pose_aa.shape[0], 6))], dim=1
                ).reshape(-1, 24, 3)
            elif pose_aa.shape[1] == 72:
                pose_aa = pose_aa.reshape(-1, 24, 3)

            B, J, N = pose_aa.shape

            if config.randomize_heading:
                random_rot = np.zeros(3)
                random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
                random_heading_rot = sRot.from_euler("xyz", random_rot)
                pose_aa[:, 0, :] = torch.tensor(
                    (
                        random_heading_rot * sRot.from_rotvec(pose_aa[:, 0, :])
                    ).as_rotvec()
                )
                trans = torch.matmul(
                    trans.float(),
                    torch.from_numpy(random_heading_rot.as_matrix().T).float(),
                )

            trans, trans_fix = self.fix_trans_height(
                pose_aa,
                trans
            )

            self.fk_model.update_model(betas=torch.zeros((1,10)), dt = 1/fps)

            fk_motion = self.fk_model.fk_batch(
                pose_aa[None,],
                trans[None,],
            )

            fk_motion = EasyDict(
                {k: v[0] if torch.is_tensor(v) else v for k, v in fk_motion.items()}
            )

            fk_motion.pose_aa = pose_aa
            res[curr_id] = fk_motion

        if queue is not None:
            queue.put(res)
        else:
            return res
            
    def get_motion_state_intervaled(
            self,
            motion_ids,
            motion_times,
            offset=None
    ):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        frame_idx = ((1.0 - blend) * frame_idx0 + blend * frame_idx1).astype(int)
        fl = frame_idx + self.length_starts[motion_ids]

        dof_pos = self.dof_pos[fl]
        body_vel = self.gvs[fl]
        body_ang_vel = self.gavs[fl]
        xpos = self.gts[fl, :]
        xquat = self.grs[fl]
        dof_vel = self.dvs[fl]
        qpos = self.qpos[fl]
        qvel = self.qvel[fl]

        if offset is not None:
            dof_pos = dof_pos + offset[..., None, :]

        return EasyDict(
            {
                "root_pos": xpos[..., 0, :].copy(),
                "root_rot": xquat[..., 0, :].copy(),
                "dof_pos": dof_pos.copy(),
                "root_vel": body_vel[..., 0, :].copy(),
                "root_ang_vel": body_ang_vel[..., 0, :].copy(),
                "dof_vel": dof_vel.reshape(dof_vel.shape[0], -1),
                "motion_aa": self._motion_aa[fl],
                "xpos": xpos,
                "xquat": xquat,
                "body_vel": body_vel,
                "body_ang_vel": body_ang_vel,
                # "motion_bodies": self._motion_bodies[motion_ids],
                "qpos": qpos,
                "qvel": qvel,
            }
        )

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def num_all_motions(self) -> int:
        """
        Returns the total number of motions in the dataset.

        Args:
            None

        Returns:
            The total number of motions in the dataset.
        """
        return self._num_unique_motions
    
    def fix_trans_height(
            self,
            pose_aa: torch.Tensor,
            trans: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            frame_check = 30
            mesh_parser = self.fk_model.smpl_parser
            vertices_curr, _ = mesh_parser.get_joints_verts(
                pose_aa[:frame_check], th_trans=trans[:frame_check]
            )

            diff_fix = vertices_curr[:frame_check, ..., -1].min(dim=-1).values.min()

            trans[..., -1] -= diff_fix

            return trans, diff_fix
        
    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.copy()
        phase = time / len
        phase = np.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0
        frame_idx0 = phase * (num_frames - 1)
        frame_idx1 = np.minimum(frame_idx0 + 1, num_frames - 1)

        blend = np.clip(
            (time - frame_idx0 * dt) / dt, 0.0, 1.0
        )  # clip blend to be within 0 and 1
        return frame_idx0, frame_idx1, blend
    
    def sample_motions(self, n=1):
        # breakpoint()
        motion_ids = np.random.choice(
            np.arange(len(self._curr_motion_ids)),
            size=n,
            p=self._sampling_batch_prob,
            replace=True,
        )
        return motion_ids