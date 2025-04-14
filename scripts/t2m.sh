#!/bin/bash

# Parse the argument
motion_file=${1}

# Run the script
python src/run.py exp_name=kinesis-moe-imitation \
    epoch=-1 \
    run=t2m \
    run.motion_file=${motion_file} \
    env.termination_distance=0.5 \