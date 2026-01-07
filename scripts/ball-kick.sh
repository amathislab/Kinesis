#!/bin/bash

python src/run.py \
    --config-name config_legs_back.yaml \
    exp_name=ball-kick \
    run=eval_run_legs_back \
    env=ball_reach \
    run.xml_path="data/xml/legs_back/leg_soccer/myolegs_soccer_goalie.xml" \
    learning=ball_reach \
    learning.actor_type=lattice \
    run.control_mode=direct \
    epoch=-1 \
    run.headless=False \
    run.num_motions=0