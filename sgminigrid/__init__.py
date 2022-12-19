from __future__ import annotations

from gymnasium.envs.registration import register

from minigrid import minigrid_env, wrappers
from minigrid.core import roomgrid

def register_sgminigrid_envs():
    register(
        id="SGMG-Button-v0",
        entry_point="sgminigrid.button_env:ButtonEnv"
    )
