#!/bin/bash

# default values
model=legs
dataset=test
headless=False
actor_type=moe
exp_name=kinesis-moe-imitation

# parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            model=$2
            shift
            shift
            ;;
        --exp_name)
            exp_name=$2
            shift
            shift
            ;;
        --actor_type)
            actor_type=$2
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
    initial_pose_dir="data/initial_pose/legs"
elif [[ $model == "legs_abs" ]]; then
    echo "Using legs_abs model"
    config_name="config_legs_abs.yaml"
    run_config="eval_run_legs_abs"
    initial_pose_dir="data/initial_pose/legs_abs"
elif [[ $model == "legs_back" ]]; then
    echo "Using legs_back model"
    config_name="config_legs_back.yaml"
    run_config="eval_run_legs_back"
    initial_pose_dir="data/initial_pose/legs_back"
# elif [[ $model == "fullbody" ]]; then
#     echo "Using fullbody model"
#     config_name="config_fullbody.yaml"
#     run_config="eval_run_fullbody"
#     initial_pose_dir="data/initial_pose/fullbody"
else
    echo "Invalid model: $model. Currently only 'legs', 'legs_abs', and 'legs_back' models are supported."
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

if [[ $actor_type == "moe" ]]; then
    actor_type=moe
elif [[ $actor_type == "lattice" ]]; then
    actor_type=lattice
else
    echo "Invalid actor type: $actor_type. Use 'moe' or 'lattice'."
    exit 1
fi

# Run the script
python src/run.py \
    --config-name ${config_name} \
    exp_name=${exp_name} \
    epoch=-1 \
    learning.actor_type=${actor_type} \
    run=${run_config} \
    run.headless=${headless} \
    run.motion_file=${motion_file} \
    run.initial_pose_file=${initial_pose_file} \
    env.termination_distance=0.5 \