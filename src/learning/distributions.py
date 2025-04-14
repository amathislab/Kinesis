# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.
#
# 2. PyTorch-RL (https://github.com/Khrylx/PyTorch-RL)
#   Copyright (c) 2020 Ye Yuan

from torch.distributions import Normal

class DiagGaussian(Normal):

    def __init__(self, loc, scale):
        super().__init__(loc, scale)

    def kl(self):
        loc1 = self.loc
        scale1 = self.scale
        log_scale1 = self.scale.log()
        loc0 = self.loc.detach()
        scale0 = self.scale.detach()
        log_scale0 = log_scale1.detach()
        kl = log_scale1 - log_scale0 + (scale0.pow(2) + (loc0 - loc1).pow(2)) / (2.0 * scale1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def log_prob(self, value):
        return super().log_prob(value).sum(1, keepdim=True)

    def mean_sample(self):
        return self.loc
