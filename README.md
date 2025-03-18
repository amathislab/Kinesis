# KINESIS: Reinforcement Learning-Based Motion Imitation for Physiologically Plausible Musculoskeletal Motor Control

<p align="center">
  <img src="./assets/Fig1-abstract.png" alt="KINESIS Logo" width="400"/>
</p>

## Overview

KINESIS is a model-free reinforcement learning framework for physiologically plausible musculoskeletal motor control. Using a musculoskeletal model of the lower body with 80 muscle actuators and 20 degrees of freedom, KINESIS achieves strong imitation performance on motion capture data, is controllable via natural language, and can be fine-tuned for high-level tasks like goal-directed movement.

Importantly, KINESIS generates muscle activity patterns that correlate well with human electromyography (EMG) data, making it a promising model for tackling challenging problems in human motor control theory.

> **Note:** The code and datasets will be available soon!

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

Instructions for installation will be provided once the code is released.

## Usage

Example usage scripts will be available with the code release.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{simos2025kinesis,
  title={Reinforcement learning-based motion imitation for physiologically plausible musculoskeletal motor control},
  author={Simos, Merkourios and Chiappa, Alberto Silvio and Mathis, Alexander},
  journal={},
  year={2025}
}
```

## Acknowledgements

This project was funded by Swiss SNF grant (310030 212516). We thank members of the Mathis Group for helpful feedback.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.