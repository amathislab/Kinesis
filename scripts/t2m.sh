#!/bin/bash

# Defaul arguments
motion_file=data/t2m/mdm_turn_left_0.pkl
model=legs
headless=False

# parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            model=$2
            shift
            shift
            ;;
        --motion_file)
            motion_file=$2
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
elif [[ $model == "legs_abs" ]]; then
    echo "Using legs_abs model"
    config_name="config_legs_abs.yaml"
    run_config="eval_run_legs_abs"
elif [[ $model == "legs_back" ]]; then
    echo "Using legs_back model"
    config_name="config_legs_back.yaml"
    run_config="eval_run_legs_back"
# elif [[ $model == "fullbody" ]]; then
#     echo "Using fullbody model"
#     config_name="config_fullbody.yaml"
#     run_config="eval_run_fullbody"
else
    echo "Invalid model: $model. Currently only 'legs', 'legs_abs', and 'legs_back' models are supported."
    exit 1
fi

# Run the script
python src/run.py \
    --config-name ${config_name} \
    exp_name=kinesis-moe-imitation \
    epoch=-1 \
    run=${run_config} \
    run.motion_file=${motion_file} \
    run.num_motions=1 \
    run.im_eval=False \
    run.headless=${headless} \
    env.termination_distance=0.5 \