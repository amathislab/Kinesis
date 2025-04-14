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

import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sys.path.append(os.getcwd())

import torch
import numpy as np
import wandb

from src.agents import agent_dict
from omegaconf import DictConfig, OmegaConf

import hydra


@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    print(
        f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )
    cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if (not cfg.no_log) and (not cfg.run.test):
        group = cfg.get("group", cfg.learning.agent_name)
        wandb.init(
            project=cfg.project,
            group=group,
            resume=not cfg.resume_str is None,
            id=cfg.resume_str,
            notes=cfg.notes,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
        )
        wandb.run.name = cfg.exp_name
        wandb.run.save()

        wandb.log({"config": OmegaConf.to_container(cfg, resolve=True)})

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Using: {device}, setting to deterministic")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # breakpoint()
    agent = agent_dict[cfg.learning.agent_name](
        cfg, dtype, device, training=True, checkpoint_epoch=cfg.epoch
    )

    if cfg.run.test:
        if cfg.run.im_eval:
            agent.eval_policy(epoch=cfg.epoch)
        else:
            cfg.num_threads = 1
            agent.run_policy()
    else:
        # breakpoint()
        agent.optimize_policy()
        print("training done!")


if __name__ == "__main__":
    main()
