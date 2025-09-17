#!/bin/bash

python src/run.py exp_name=lattice_direct_target_reach \
    run=eval_run \
    learning=pointgoal \
    epoch=-1 \
    run.headless=True \
    run.control_mode=direct