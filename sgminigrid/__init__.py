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
        id="SGMG-BDoor-v0",
        entry_point="sgminigrid.envs.buttondoor:ButtonDoorEnv",
        kwargs={"size": 7, "num_colors": 4, "num_extra_buttons": 3, "max_steps": 100, "generalize": False}
    )
    register(
        id="SGMG-BDoorL-v0",
        entry_point="sgminigrid.envs.buttondoor:ButtonDoorEnv",
        kwargs={"size": 7, "num_colors": 6, "num_extra_buttons": 4, "max_steps": 100, "generalize": False}
    )
    register(
        id="SGMG-BDoorF-v0",
        entry_point="sgminigrid.envs.buttondoorfull:ButtonDoorFullEnv",
        kwargs={"size": 7, "num_colors": 4, "num_extra_buttons": 3, "max_steps": 100, "generalize": False}
    )
    register(
        id="SGMG-BDoorG-v0",
        entry_point="sgminigrid.envs.buttondoor:ButtonDoorEnv",
        kwargs={"size": 7, "num_colors": 4, "num_extra_buttons": 3, "max_steps": 100, "generalize": True}
    )
    register(
        id="SGMG-BDoorFG-v0",
        entry_point="sgminigrid.envs.buttondoorfull:ButtonDoorFullEnv",
        kwargs={"size": 7, "num_colors": 4, "num_extra_buttons": 3, "max_steps": 100, "generalize": True}
    )
    register(
        id="SGMG-BDoorFCG-v0",
        entry_point="sgminigrid.envs.buttondoorfull:ButtonDoorFullEnv",
        kwargs={"size": 7, "num_colors": 4, "num_extra_buttons": 3, "max_steps": 100, "generalize": True, "gen_class": "color"}
    )
    register(
        id="SGMG-BDoorLFG-v0",
        entry_point="sgminigrid.envs.buttondoorfull:ButtonDoorFullEnv",
        kwargs={"size": 7, "num_colors": 6, "num_extra_buttons": 5, "max_steps": 100, "generalize": True}
    )
    register(
        id="SGMG-BDoorLFCG-v0",
        entry_point="sgminigrid.envs.buttondoorfull:ButtonDoorFullEnv",
        kwargs={"size": 7, "num_colors": 6, "num_extra_buttons": 5, "max_steps": 100, "generalize": True, "gen_class": "color"}
    )
    register(
        id="SGMG-BDoorCG-v0",
        entry_point="sgminigrid.envs.buttondoor:ButtonDoorEnv",
        kwargs={"size": 7, "num_colors": 4, "num_extra_buttons": 3, "max_steps": 100, "generalize": True, "gen_class": "color"}
    )
    register(
        id="SGMG-BDoorLCG-v0",
        entry_point="sgminigrid.envs.buttondoor:ButtonDoorEnv",
        kwargs={"size": 7, "num_colors": 6, "num_extra_buttons": 5, "max_steps": 100, "generalize": True, "gen_class": "color"}
    )
    register(
        id="SGMG-Empty-v0",
        entry_point="sgminigrid.envs.empty:SGEmptyEnv",
        kwargs={"size": 8, "max_steps": 50}
    )
