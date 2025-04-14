#!/bin/bash

python src/run.py exp_name=kinesis-target-goal-reach \
    run=eval_run \
    learning=directional \
    epoch=-1 \
    run.headless=False \
    run.im_eval=False \
