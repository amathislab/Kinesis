#!/bin/bash

# default values
model=legs
dataset=test
headless=False

# parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            model=$2
            shift
            shift
            ;;
        --dataset)
            dataset=$2
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

if [[ $model == "legs" ]]; then
    echo "Using legs model"
    config_name="config_legs.yaml"
    run_config="eval_run_legs"
    initial_pose_dir="data/initial_pose/legs_model"
else
    echo "Invalid model: $model. Currently only 'legs' model is supported."
    exit 1
fi

if [[ $dataset == "train" ]]; then
    motion_file="data/kit_train_motion_dict.pkl"
    initial_pose_file="${initial_pose_dir}/initial_pose_train.pkl"
elif [[ $dataset == "test" ]]; then
    motion_file="data/kit_test_motion_dict.pkl"
    initial_pose_file="${initial_pose_dir}/initial_pose_test.pkl"
else
    echo "Invalid dataset: $dataset. Use 'train' or 'test'."
    exit 1
fi

# Run the script
python src/run.py \
    --config-name ${config_name} \
    exp_name=kinesis-moe-imitation \
    epoch=-1 \
    run=${run_config} \
    run.headless=${headless} \
    run.motion_file=${motion_file} \
    run.initial_pose_file=${initial_pose_file} \
    env.termination_distance=0.5 \