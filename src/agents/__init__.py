# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# This file contains code adapted from:
#
# 1. PHC_MJX (https://github.com/ZhengyiLuo/PHC_MJX)

from .agent_im import AgentIM
from .agent import Agent
from .agent_pg import AgentPG
from .agent_ppo import AgentPPO
from .agent_humanoid import AgentHumanoid
from .agent_pointgoal import AgentPointGoal
from .agent_directional import AgentDirectional

agent_dict = {
    'agent': Agent,
    'agent_pg': AgentPG,
    'agent_ppo': AgentPPO,   
    'agent_humanoid': AgentHumanoid,
    'agent_im': AgentIM,
    'agent_pointgoal': AgentPointGoal,
    'agent_directional': AgentDirectional
}