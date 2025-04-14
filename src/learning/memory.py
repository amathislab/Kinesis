# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.
#
# 2. PyTorch-RL (https://github.com/Khrylx/PyTorch-RL)
#   Copyright (c) 2020 Ye Yuan

import random
random.seed(0)


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a tuple."""
        self.memory.append([*args])

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.memory
        else:
            random_batch = random.sample(self.memory, batch_size)
            return random_batch

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)
