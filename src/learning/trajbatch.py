# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.
#
# 2. PyTorch-RL (https://github.com/Khrylx/PyTorch-RL)
#   Copyright (c) 2020 Ye Yuan

import numpy as np

class TrajBatch:
    def __init__(self, memory_list):
        memory = memory_list[0]
        for x in memory_list[1:]:
            memory.append(x)
        self.batch = zip(*memory.sample())
        self.states = np.stack(next(self.batch))
        self.actions = np.stack(next(self.batch))
        self.masks = np.stack(next(self.batch))
        self.next_states = np.stack(next(self.batch))
        self.rewards = np.stack(next(self.batch))
        self.exps = np.stack(next(self.batch))
