#!/bin/bash

python src/run.py \
    --config-name config_legs.yaml \
    exp_name=target-reach \
    run=eval_run_legs \
    learning=directional \
    epoch=-1 \
    run.headless=False \
    run.im_eval=False \
