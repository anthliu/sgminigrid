from __future__ import annotations

from gymnasium.envs.registration import register

from minigrid import minigrid_env, wrappers
from minigrid.core import roomgrid

def register_sgminigrid_envs():
    register(
        id="SGMG-ButtonTest-v0",
        entry_point="sgminigrid.envs.buttontest:ButtonEnv"
    )
    register(
        id="SGMG-ButtonDoor-v0",
        entry_point="sgminigrid.envs.buttondoor:ButtonDoorEnv",
        kwargs={"size": 7, "num_colors": 4, "num_extra_buttons": 3, "max_steps": 100}
    )
    register(
        id="SGMG-ButtonDoorL-v0",
        entry_point="sgminigrid.envs.buttondoor:ButtonDoorEnv",
        kwargs={"size": 7, "num_colors": 6, "num_extra_buttons": 4, "max_steps": 100}
    )
    register(
        id="SGMG-Empty-v0",
        entry_point="sgminigrid.envs.empty:SGEmptyEnv",
        kwargs={"size": 8, "max_steps": 50}
    )
