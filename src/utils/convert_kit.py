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

import glob
import os
import sys
import os.path as osp
sys.path.append(os.getcwd())

import torch 
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import joblib
from tqdm import tqdm
import argparse
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from src.utils.smpl_skeleton.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from src.utils.smpl_skeleton.smpl_local_robot import SMPL_Robot as LocalRobot

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="")
    args = parser.parse_args()
    
    upright_start = True
    robot_cfg = {
            "mesh": False,
            "rel_joint_lm": True,
            "upright_start": upright_start,
            "remove_toe": False,
            "real_weight": True,
            "real_weight_porpotion_capsules": True,
            "real_weight_porpotion_boxes": True, 
            "replace_feet": True,
            "masterfoot": False,
            "big_ankle": True,
            "freeze_hand": False, 
            "box_body": False,
            "master_range": 50,
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
            "model": "smpl",
        }

    smpl_local_robot = LocalRobot(robot_cfg,)
    if not osp.isdir(args.path):
        print("Please specify AMASS data path")
        exit(-1)
        
    all_pkls = glob.glob(f"{args.path}/**/*.npz", recursive=True)
    amass_full_motion_dict = {}

    with open("data/kit_train_keys.txt", "r") as f:
        train_keys = f.readlines()
    train_keys = [key.strip() for key in train_keys]

    with open("data/kit_test_keys.txt", "r") as f:
        test_keys = f.readlines()
    test_keys = [key.strip() for key in test_keys]

    length_acc = []
    for data_path in tqdm(all_pkls):
        bound = 0
        splits = data_path.split("/")[-2:]
        key_name_dump = "0-KIT_" + "_".join(splits).replace(".npz", "")
        
        if key_name_dump not in train_keys and key_name_dump not in test_keys:
            continue
            
        entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
        
        if not 'mocap_framerate' in  entry_data:
            continue
        framerate = entry_data['mocap_framerate']
        
        skip = int(framerate/30)
        root_trans = entry_data['trans'][::skip, :]
        pose_aa = np.concatenate([entry_data['poses'][::skip, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
        betas = entry_data['betas']
        gender = entry_data['gender']
        N = pose_aa.shape[0]
        
        if bound == 0:
            bound = N
            
        root_trans = root_trans[:bound]
        pose_aa = pose_aa[:bound]
        N = pose_aa.shape[0]
        if N < 10:
            continue
    
        smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
        pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

        beta = np.zeros((16))
        gender_number, beta[:], gender = [0], 0, "neutral"

        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
        smpl_local_robot.write_xml(f"data/xml/{robot_cfg['model']}_humanoid.xml")
        skeleton_tree = SkeletonTree.from_mjcf(f"data/xml/{robot_cfg['model']}_humanoid.xml")
        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
                    torch.from_numpy(pose_quat),
                    root_trans_offset,
                    is_local=True)
        
        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)  # should fix pose_quat as well here...

            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
            pose_quat = new_sk_state.local_rotation.numpy()


        pose_quat_global = new_sk_state.global_rotation.numpy()
        pose_quat = new_sk_state.local_rotation.numpy()
        fps = 30

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

    kit_train_motion_dict = {
        key: amass_full_motion_dict[key] for key in train_keys
    }

    kit_test_motion_dict = {
        key: amass_full_motion_dict[key] for key in test_keys
    }

    joblib.dump(kit_train_motion_dict, "data/kit_train_motion_dict.pkl")
    joblib.dump(kit_test_motion_dict, "data/kit_test_motion_dict.pkl")