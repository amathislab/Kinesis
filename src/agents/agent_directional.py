# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.

import os
import logging

os.environ["OMP_NUM_THREADS"] = "1"

from src.agents.agent_im import AgentIM
from src.env.myolegs_directional_control import MyoLegsDirectional

logger = logging.getLogger(__name__)

class AgentDirectional(AgentIM):
    """
    AgentDirectional.
    """

    def __init__(self, cfg, dtype, device, training = True, checkpoint_epoch = 0):
        super().__init__(cfg, dtype, device, training, checkpoint_epoch)

    def setup_env(self):
        self.env = MyoLegsDirectional(self.cfg)
        logger.info("Directional control environment initialized.")