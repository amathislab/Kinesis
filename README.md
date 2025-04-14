[![Downloads](https://img.shields.io/badge/dynamic/json?url=https://huggingface.co/api/datasets/amathislab/kinesis-assets&query=downloads&label=downloads)](https://huggingface.co/papers/2503.14637)

# KINESIS: Reinforcement Learning-Based Motion Imitation for Physiologically Plausible Musculoskeletal Motor Control

<p align="center">
  <img src="./assets/Fig1-abstract.png" alt="KINESIS Logo" width="400"/>
</p>

## Overview

KINESIS is a model-free reinforcement learning framework for physiologically plausible musculoskeletal motor control. Using a musculoskeletal model of the lower body with 80 muscle actuators and 20 degrees of freedom, KINESIS achieves strong imitation performance on motion capture data, is controllable via natural language, and can be fine-tuned for high-level tasks like goal-directed movement.

Importantly, KINESIS generates muscle activity patterns that correlate well with human electromyography (EMG) data, making it a promising model for tackling challenging problems in human motor control theory. Check out the [arxiv article for more details!](https://arxiv.org/abs/2503.14637)

## Demonstrations

<h3 align="center">Motion Imitation</h3>
<p align="center">
  <img src="./assets/kit/kit_walk.gif" alt="Walking Forward" width="19%"/>
  <img src="./assets/kit/kit_gradual_turn.gif" alt="Gradual Turn" width="19%"/>
  <img src="./assets/kit/kit_turn_in_place.gif" alt="Turn in Place" width="19%"/>
  <img src="./assets/kit/kit_backward.gif" alt="Walking Backwards" width="19%"/>
  <img src="./assets/kit/kit_run.gif" alt="Running" width="19%"/>
</p>

<h3 align="center">Text-to-Motion Control</h3>
<p align="center">
  <img src="./assets/t2m/t2m_forward.gif" alt="Walking Forward" width="19%"/>
  <img src="./assets/t2m/t2m_circle.gif" alt="Circling" width="19%"/>
  <img src="./assets/t2m/t2m_backwards.gif" alt="Walking Backwards" width="19%"/>
  <img src="./assets/t2m/t2m_left.gif" alt="Turning Left" width="19%"/>
  <img src="./assets/t2m/t2m_right.gif" alt="Turning Right" width="19%"/>
</p>

<h3 align="center">High-Level Control</h3>
<p align="center">
  <img src="./assets/high_level/hl_target_reach.gif" alt="Reaching Target" width="49%"/>
  <img src="./assets/high_level/hl_directional.gif" alt="Directional Control" width="49%"/>

## EMG Comparison
<p align="center">
  <img src="./assets/Fig7-emg.png" alt="EMG Comparison" width="80%"/>


## Installation

### Installing the environment
For Linux/CUDA:
```bash
conda create -n kinesis python=3.8
conda activate kinesis
pip install -r requirements.txt
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```
For MacOS:
```bash
conda create -n kinesis python=3.8
conda activate kinesis
pip install -r macos_requirements.txt
conda install -c conda-forge lxml
```

### Downloading the SMPL model
- Download the SMPL parameters from [SMPL](https://smpl.is.tue.mpg.de/) -- only the neutral body parameters are required (it's the middle link under "Download").
- Rename the file containing the neutral body parameters to `SMPL_NEUTRAL.pkl`.
- Place the file in the `data/smpl` directory.

### Downloading and processing KIT
- First, download the KIT dataset from the AMASS website (https://amass.is.tue.mpg.de -- it's the SMPL-H dataset).
- Then, run the following script on the unzipped directory:
```bash
python src/utils/convert_kit.py --path <path_to_kit_dataset>
```

### Downloading assets
- Run the following script to download the assets from Hugging Face:
```bash
pip install huggingface_hub
python src/utils/download_assets.py
```

### Downloading the pre-trained models
- Run the following scripts to download the models from Hugging Face:
```bash
python src/utils/download_model.py --repo_id amathislab/kinesis-moe-imitation
python src/utils/download_model.py --repo_id amathislab/kinesis-target-goal-reach
```
- The saved `model.pth` checkpoints will be saved in the `data/trained_models` directory.

## Usage
> **Note:** For MacOS users, you will need to change the command in the bash scripts to `mjpython` instead of `python`.

### KIT-Locomotion imitation
To test Kinesis on motion imitation, run the following command:
```bash
# Train set
bash scripts/kit-locomotion.sh --mode train
# Test set
bash scripts/kit-locomotion.sh --mode test
```
You can turn rendering on/off by setting the `--headless` flag to `False` or `True`, respectively.

### Text-to-Motion Control
To test Kinesis on text-to-motion control, select one of the pre-generated motions in the `data/t2m` directory and run the following command:
```bash
bash scripts/t2m.sh <motion_path>
```

### Target Reaching
To test Kinesis on target reaching, run the following command:
```bash
bash scripts/target-reach.sh
```

### Directional Control
To test Kinesis on directional control, run the following command:
```bash
bash scripts/directional.sh
```
By default, you can control the direction of the motion using the numpad keys on your keyboard. If your keyboard does not have a numpad, you can define the keymaps in the `src/env/myolegs_directional_control.py` file:
```python
...
def key_callback(self, keycode):
        super().key_callback(keycode)
        if keycode == YOUR_KEYCODE_NORTHEAST:
            # Point the goal to north-east
            self.stop = False
            self.direction = np.pi / 4
        elif keycode == YOUR_KEYCODE_NORTH:
            # Point the goal to north
            self.stop = False
            self.direction = np.pi / 2
...
```

## Training a new policy
Kinesis consists of three policy experts, combined with a Mixture of Experts (MoE) module. Training is done iteratively through negative mining: First, we train the first expert on the training set, then we train the second expert on only the samples that the first expert failed to imitate, and so on until we reach sufficient performance. Finally, we train the MoE module to combine the experts, which are frozen during this step. We use Weights & Biases to log the training process.

#### Training single experts through negative mining

Training an expert from scratch is done by running the following command:
```bash
python src/run.py project=<project_name> exp_name=<experiment_name> epoch=-1 run=train_run run.num_threads=<num_threads> learning.actor_type="lattice"
```
- `<project_name>`: The name of the project for logging purposes.
- `<experiment_name>`: The name of the experiment for logging purposes.
- `<num_threads>`: The number of threads to use for training. We usually set this to the amount of CPU threads available on the machine.

Once the policy has reached a plateau (in terms of average episode length), we need to evaluate its performance on the **train** set, and identify the samples that the policy fails to imitate. This is done by running the following command:
```bash
python src/utils/save_failed_motions.py --exp_name <experiment_name> --epoch <epoch> --expert 0
```
- `<experiment_name>`: The name of the experiment under which the policy was trained.
- `<epoch>`: The epoch at which to evaluate the policy. Make sure that there exists a saved checkpoint at this epoch.
- `--expert`: The expert to evaluate. This should be set to 0 for the first expert, 1 for the second expert, and so on.

The failed samples will be saved as a new training set with the name `kit_train_motion_dict_expert_<X>`, where `<X>` is the expert number, along with the respective initial pose data.

To train the next expert, run the following command:
```bash
python src/run.py project=<project_name> exp_name=<experiment_name>_expert_1 epoch=-1 run=train_run run.num_threads=<num_threads> learning.actor_type="lattice" run.motion_file=data/kit_train_motion_dict_expert_1 run.initial_pose_file=data/kit_train_initial_pose_expert_1.pkl
```

... and so on, until the negative-mined dataset is empty or sufficiently small.

#### Training the Mixture of Experts (MoE) module
Once all experts have been trained, we can train the MoE module to combine them. To train the MoE module, run the following commands:
```bash
# Rename the initial experiment folder:
mv data/trained_models/<experiment_name> data/trained_models/<experiment_name>_expert_0
# Train the MoE module:
python src/run.py project=<project_name> exp_name=<experiment_name>_moe epoch=0 run=train_run run.expert_path=data/trained_models/<experiment_name> run.num_threads=<num_threads> learning.actor_type="moe"
```

## Citation

If you find this work useful in your research, please consider citing our [paper](https://arxiv.org/abs/2503.14637):

```bibtex
@article{simos2025kinesis,
  title={Reinforcement learning-based motion imitation for physiologically plausible musculoskeletal motor control},
  author={Simos, Merkourios and Chiappa, Alberto Silvio and Mathis, Alexander},
  journal={arXiv},
  year={2025},
  doi={10.48550/arXiv.2503.14637}
}
```

## Acknowledgements

This work would not have been possible without the amazing contributions of [PHC](https://github.com/ZhengyiLuo/PHC), [PHC_MJX](https://github.com/ZhengyiLuo/PHC_MJX), [SMPLSim](https://github.com/ZhengyiLuo/SMPLSim), on which the code is based, as well as [MyoSuite](https://sites.google.com/view/myosuite), from which we borrow the musculoskeletal model. Please consider citing & starring these repositories if you find them useful!

For text-to-motion generation, we used the awesome work from Tevet et al. -- [Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model). Please consider citing their work if you use the text-to-motion generation code.

We also acknowledge the foundational contribution of the Max Planck Institute for Intelligent Systems, which developed the [SMPL model](https://smpl.is.tue.mpg.de/) and curated the [AMASS dataset](https://amass.is.tue.mpg.de/).

The EMG analysis uses processed data from the paper "A wearable real-time kinetic measurement sensor setup for human locomotion" by [Wang et al. (2023)](https://www.cambridge.org/core/journals/wearable-technologies/article/wearable-realtime-kinetic-measurement-sensor-setup-for-human-locomotion/488C21B7706FFDFA7FFAB387FD0A1A64?utm_campaign=shareaholic&utm_medium=copy_link&utm_source=bookmark). We thank the authors for creating this open-source dataset. Please cite the original paper if you use the EMG data.

This project was funded by Swiss SNF grant (310030 212516). We thank members of the Mathis Group for helpful feedback.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
