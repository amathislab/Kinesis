#!/bin/bash

python src/run.py exp_name=kinesis-target-goal-reach \
    run=eval_run \
    learning=pointgoal \
    epoch=-1 \
    run.headless=False \