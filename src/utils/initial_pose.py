# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.

import sys
import os
import joblib
sys.path.append(os.getcwd())

from src.env.myolegs_im import MyoLegsIm

import hydra
import numpy as np

@hydra.main(
    version_base=None,
    config_path="../data/cfg",
    config_name="config",
)
def main(cfg):
    initial_pose_dict = {}
    env = MyoLegsIm(cfg)
    env.initial_pos_data = {}
    
    for motion_step in range(cfg.run.num_motions):
        env.motion_lib.load_motions(
            env.motion_lib_cfg,
            shape_params=env.gender_betas,
            random_sample=False,
            start_idx=motion_step,
        )
        motion_length = env.motion_lib._motion_lengths[0]
        initial_pose_dict[env.motion_lib._curr_motion_ids[0]] = {}
        print(f'Processing motion {env.motion_lib._curr_motion_ids[0]}: {motion_length} frames')
        for start_time in np.arange(0, motion_length, 0.2):
            print(f'Start time: {start_time}')
            env.reset(options={'start_time': start_time})
            initial_pose_dict[env.motion_lib._curr_motion_ids[0]][start_time] = env.initial_pose
            env.initial_pose = None

    # Fix the keys
    new_data = {}
    for motion_key in initial_pose_dict.keys():
        new_data[motion_key] = {}
        for frame_key in initial_pose_dict[motion_key].keys():
            new_key = np.round(frame_key, 1)
            new_data[motion_key][new_key] = initial_pose_dict[motion_key][frame_key]
            print(f'Old key: {frame_key}, New key: {new_key}')

    joblib.dump(new_data, f'data/initial_pose/initial_pose_{cfg.exp_name}.pkl')

if __name__ == "__main__":
    main()