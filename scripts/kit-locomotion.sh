#!/bin/bash

# default values
mode=test
headless=False

# parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            mode=$2
            shift
            shift
            ;;
        --headless)
            render=$2
            shift
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ $mode == "train" ]]; then
    motion_file="data/kit_train_motion_dict.pkl"
    initial_pose_file="data/initial_pose/initial_pose_train.pkl"
elif [[ $mode == "test" ]]; then
    motion_file="data/kit_test_motion_dict.pkl"
    initial_pose_file="data/initial_pose/initial_pose_test.pkl"
else
    echo "Invalid mode: $mode. Use 'train' or 'test'."
    exit 1
fi

# Run the script
python src/run.py exp_name=kinesis-moe-imitation \
    epoch=-1 \
    run=eval_run \
    run.headless=${headless} \
    run.motion_file=${motion_file} \
    run.initial_pose_file=${initial_pose_file} \
    env.termination_distance=0.5 \