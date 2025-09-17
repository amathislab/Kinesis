#!/bin/bash

# default values
mode=test
headless=True

# parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            mode=$2
            shift
            shift
            ;;
        --headless)
            headless=$2
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
    motion_file="data/kit_train_motion_dict_new.pkl"
    initial_pose_file="data/initial_pose/initial_pose_train_new.pkl"
elif [[ $mode == "test" ]]; then
    motion_file="data/kit_test_motion_dict_new.pkl"
    initial_pose_file="data/initial_pose/initial_pose_test_new.pkl"
else
    echo "Invalid mode: $mode. Use 'train' or 'test'."
    exit 1
fi

# Run the script
python src/run.py exp_name=lattice_direct_moe \
    epoch=3000 \
    run=eval_run \
    run.headless=${headless} \
    run.motion_file=${motion_file} \
    run.initial_pose_file=${initial_pose_file} \
    env.termination_distance=0.5 \
    run.control_mode=direct \
    run.recording_biomechanics=False \
    seed=0