# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# This file contains code adapted from:
#
# 1. PHC (https://github.com/ZhengyiLuo/PHC)
#    Copyright (c) 2023 Carnegie Mellon University
#    Copyright (c) 2018-2023, NVIDIA Corporation
#    All rights reserved.

from ast import Try
import torch
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import os
import sys
import os.path as osp

sys.path.append(os.getcwd())

from src.smpl.smpl_parser import SMPL_BONE_ORDER_NAMES as joint_names
from src.utils.smpl_skeleton.smpl_local_robot import SMPL_Robot as LocalRobot
import scipy.ndimage.filters as filters
from tqdm import tqdm
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

import argparse

def quat_correct(quat):
    """ Converts quaternion to minimize Euclidean distance from previous quaternion (wxyz order) """
    for q in range(1, quat.shape[0]):
        if np.linalg.norm(quat[q - 1] - quat[q], axis=0) > np.linalg.norm(quat[q - 1] + quat[q], axis=0):
            quat[q] = -quat[q]
    return quat

robot_cfg = {
    "mesh": False,
    "model": "smpl",
    "upright_start": True,
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
}
print(robot_cfg)

smpl_local_robot = LocalRobot(
    robot_cfg,
    data_dir="data/smpl",
)

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", 
                    type=str, 
                    help="Path to the input file (.pkl) containing the SMPL parameters including pose_aa")
parser.add_argument("--output_dir",
                    type=str,
                    help="Path to the output directory where the processed data will be saved")
parser.add_argument("--fps",
                    type=int,
                    default=20,
                    help="FPS of the motion")
params = parser.parse_args()

assert params.input_path.endswith(".pkl")

motion_name = params.input_path.split("/")[-1].split(".")[0]

res_data = joblib.load(params.input_path)

amass_data = {}

pose_aa = np.array(res_data['pose_aa'])
B = pose_aa.shape[0]
trans = np.array(res_data['root_translation']).T
pose_aa = pose_aa.reshape(B, 72)

transform = sRot.from_euler('xyz', np.array([np.pi / 2, 0, 0]), degrees=False)
new_root = (transform * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec()

pose_aa[:, :3] = new_root

trans = trans.dot(transform.as_matrix().T)

amass_data["0"] = {"pose_aa": pose_aa, "trans": trans, 'beta': np.zeros(10)}

mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

amass_full_motion_dict = {}
for key_name in tqdm(amass_data.keys()):
    key_name_dump = key_name
    smpl_data_entry = amass_data[key_name]

    pose_aa = smpl_data_entry['pose_aa'].copy()
    root_trans = smpl_data_entry['trans'].copy()
    B = pose_aa.shape[0]

    beta = smpl_data_entry['beta'].copy() if "beta" in smpl_data_entry else smpl_data_entry['betas'].copy()
    if len(beta.shape) == 2:
        beta = beta[0]

    fps = params.fps

    smpl_2_mujoco = [joint_names.index(q) for q in mujoco_joint_names if q in joint_names]
    batch_size = pose_aa.shape[0]
    pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)
    pose_aa_mj = pose_aa.reshape(-1, 24, 3)[..., smpl_2_mujoco, :].copy()

    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)

    gender_number, beta[:], gender = [0], 0, "neutral"
    print("using neutral model")

    smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
    smpl_local_robot.write_xml("smpl_humanoid_1.xml")
    skeleton_tree = SkeletonTree.from_mjcf("smpl_humanoid_1.xml")

    root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
        torch.from_numpy(pose_quat),
        root_trans_offset,
        is_local=True)

    if robot_cfg['upright_start']:
        pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)  # should fix pose_quat as well here...

        print("############### filtering!!! ###############")
        import scipy.ndimage.filters as filters
        root_trans_offset = filters.gaussian_filter1d(root_trans_offset, 3, axis=0, mode="nearest")
        root_trans_offset = torch.from_numpy(root_trans_offset)
        pose_quat_global = np.stack([quat_correct(pose_quat_global[:, i]) for i in range(pose_quat_global.shape[1])], axis=1)


        filtered_quats = filters.gaussian_filter1d(pose_quat_global, 2, axis=0, mode="nearest")
        pose_quat_global = filtered_quats / np.linalg.norm(filtered_quats, axis=-1)[..., None]
        print("############### filtering!!! ###############")
        new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
        pose_quat = new_sk_state.local_rotation.numpy()

    new_motion_out = {}
    new_motion_out['pose_quat_global'] = pose_quat_global
    new_motion_out['pose_quat'] = pose_quat
    new_motion_out['trans_orig'] = root_trans
    new_motion_out['root_trans_offset'] = root_trans_offset
    new_motion_out['beta'] = beta
    new_motion_out['gender'] = gender
    new_motion_out['pose_aa'] = pose_aa
    new_motion_out['fps'] = fps
    amass_full_motion_dict[key_name_dump] = new_motion_out

    print(pose_aa.shape)

joblib.dump(amass_full_motion_dict, osp.join(params.output_dir, f"{motion_name}.pkl"))