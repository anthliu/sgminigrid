from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from sgminigrid.sgminigrid_env import SGMiniGridEnv
from sgminigrid.sgworld_object import Button, ButtonDoor
import numpy as np


class SGTunnel(SGMiniGridEnv):
    def __init__(
        self,
        size=10,
        max_steps: int | None = None,
        compose=False,
        **kwargs,
    ):
        self.compose = compose
        if compose:
            place_holders = [['both']]
        else:
            place_holders = [['red', 'blue']]
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=place_holders
        )
        '''
        completion_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[['red', 'blue', 'both']]
        )
        '''
        completion_space = mission_space


        if max_steps is None:
            max_steps = 10 * size

        super().__init__(
            mission_space=mission_space,
            completion_space=completion_space,
            width=size,
            height=3,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str):
        if color == 'both':
            return f"press both buttons"
        else:
            return f"press the {color} button"

    def _sample_task(self, task_id=None):
        if self.compose:
            self.goal = 'both'
        else:
            if task_id is None:
                task_id = self._rand_int(0, 2)
            else:
                task_id = task_id % 2
            self.goal = ['red', 'blue'][task_id]

    def _gen_grid(self, width, height):
        self.task_infos = {}# Info about current task for logging
        self.task_infos['tags'] = []

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.red = Button('red', is_pressed=False)
        self.put_obj(self.red, 1, 1)

        self.blue1 = Button('blue', is_pressed=False)
        self.put_obj(self.blue1, 3, 1)

        self.blue2 = Button('blue', is_pressed=False)
        self.put_obj(self.blue2, width-2, 1)

        # Place the agent
        self.agent_pos = (width-4, 1)
        self.agent_dir = 0
        
        goal = self.goal
        self.task_infos['tags'].append(goal)
        self.mission = self._gen_mission(goal)

        if goal == 'red':
            self.goal_func = lambda: self.red.is_pressed
        elif goal == 'blue':
            self.goal_func = lambda: self.blue1.is_pressed or self.blue2.is_pressed
        elif goal == 'both':
            self.goal_func = lambda: self.red.is_pressed and (self.blue1.is_pressed or self.blue2.is_pressed)

    def _subtask_completions(self):
        completion = {}
        if self.compose:
            completion[self._gen_mission('both')] = self.red.is_pressed and (self.blue1.is_pressed or self.blue2.is_pressed)
        else:
            completion[self._gen_mission('red')] = self.red.is_pressed
            completion[self._gen_mission('blue')] = self.blue1.is_pressed or self.blue2.is_pressed
        return completion

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.goal_func():
            reward = self._reward()
            terminated = True
        return obs, reward, terminated, truncated, info
