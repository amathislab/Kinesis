#!/bin/bash

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

python src/run.py \
    --config-name ${config_name} \
    exp_name=target-reach \
    run=${run_config} \
    learning=pointgoal \
    epoch=-1 \
    run.headless=${headless} \