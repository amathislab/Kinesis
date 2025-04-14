# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.

import os
import sys
import argparse
import joblib

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sys.path.append(os.getcwd())

import torch
import numpy as np

from src.agents import agent_dict
from omegaconf import DictConfig

import hydra

def eval_cfg() -> DictConfig:
    """Create a config fixture for testing the evaluation performance."""
    with hydra.initialize(version_base=None, config_path="../../cfg"):
        cfg = hydra.compose(config_name="test_eval_config")
    return cfg

def eval_imitation(eval_cfg: DictConfig, exp_name: str, epoch: int = -1, expert: int = 0) -> None:
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Using: {device}, setting to deterministic")
    np.random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    eval_cfg.output_dir = f"data/trained_models/{exp_name}"
    eval_cfg.epoch = epoch
    eval_cfg.run.headless = True
    eval_cfg.learning.actor_type = "lattice"

    if expert == 0:
        eval_cfg.run.motion_file = "data/kit_train_motion_dict.pkl"
        eval_cfg.run.initial_pose_file = "data/initial_pose/initial_pose_train.pkl"
    else:
        if os.path.exists(f"data/kit_train_motion_dict_expert_{expert}.pkl") and os.path.exists(f"data/initial_pose/initial_pose_train_expert_{expert}.pkl"):
            eval_cfg.run.motion_file = f"data/kit_train_motion_dict_expert_{expert}.pkl"
            eval_cfg.run.initial_pose_file = f"data/initial_pose/initial_pose_train_expert_{expert}.pkl"
        else:
            raise FileNotFoundError(
                f"Motion file or initial pose file for expert {expert} does not exist."
            )

    agent = agent_dict[eval_cfg.learning.agent_name](
        eval_cfg, dtype, device, training=True, checkpoint_epoch=eval_cfg.epoch
    )

    mpjpe_dict, success = agent.eval_policy(epoch=eval_cfg.epoch, dump=True)

    print(f"Success rate: {success}")

    failed_keys = np.load(f"data/dumps/failed_keys_{agent.cfg.epoch}.npy")

    if len(failed_keys) == 0:
        print("No failed keys found. Exiting.")
        return

    # get the motion data and initial position data
    motion_dict = joblib.load(agent.env.motion_file)
    initial_pos_data = agent.env.initial_pos_data

    # remove the failed keys from the motion data
    negative_mined_motion_dict = {}
    for i, k in enumerate(motion_dict.keys()):
        if i in failed_keys:
            negative_mined_motion_dict[k] = motion_dict[k]

    # remove the failed keys from the initial position data and re-index
    negative_mined_initial_pos_data = {}
    i = 0
    for k in initial_pos_data.keys():
        if k in failed_keys:
            negative_mined_initial_pos_data[i] = initial_pos_data[k]
            i += 1

    # save the negative mined motion data and initial position data
    joblib.dump(
        negative_mined_motion_dict,
        f"data/kit_train_motion_dict_expert_{expert + 1}.pkl"
    )
    joblib.dump(
        negative_mined_initial_pos_data,
        f"data/initial_pose/initial_pose_train_expert_{expert + 1}.pkl"
    )

    # copy the saved checkpoint to a new folder
    if expert == 0:
        os.makedirs(f"data/trained_models/{exp_name}_expert_1", exist_ok=True)
    else:
        os.makedirs(f"data/trained_models/{exp_name[:-1]}{expert + 1}", exist_ok=True)

    if epoch == -1:
        os.system(
            f"cp data/trained_models/{exp_name}/model.pth data/trained_models/{exp_name}_expert_{expert + 1}/model.pth"
            )
    else:
        os.system(
            f"cp data/trained_models/{exp_name}/model_{epoch:08d}.pth data/trained_models/{exp_name}_expert_{expert + 1}/model.pth"
            )

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Define the experiment name and epoch")
    parser.add_argument(
        "--exp_name",
        type=str,
        help="The name of the folder where the model is saved",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=-1,
        help="The epoch of the model to be evaluated",
    )
    parser.add_argument(
        "--expert",
        type=int,
        default=0,
        help="The expert number to be evaluated",
    )
    args = parser.parse_args()

    cfg = eval_cfg()
    eval_imitation(cfg, args.exp_name, args.epoch, args.expert)