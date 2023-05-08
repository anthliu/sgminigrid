from __future__ import annotations
import types

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Ball, Wall
from sgminigrid.sgminigrid_env import SGMiniGridEnv
from sgminigrid.sgworld_object import Button, ButtonDoor
import numpy as np


class SGLoading(SGMiniGridEnv):
    def __init__(
        self,
        size=7,
        max_steps: int | None = None,
        compose=False,
        curriculum=False,
        sticky=False,
        **kwargs,
    ):
        self.curriculum = curriculum
        self.compose = compose
        self.sticky = sticky# make ball objects only pickupable if not in the goal area
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
            height=4,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str):
        if color == 'both':
            return f"load both balls into the goal"
        else:
            return f"load the {color} ball into the goal"

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

        # Sample goal
        pre_place_red = False
        pre_place_blue = False
        goal = self.goal
        if not self.compose and self.curriculum:
            if self._rand_bool():
                if goal == 'blue':
                    pre_place_red = True
                    pre_place_blue = False
                else:
                    pre_place_red = False
                    pre_place_blue = True
        self.task_infos['tags'].append(goal)
        self.mission = self._gen_mission(goal)

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Wall(), 1, 1)
        self.put_obj(Wall(), 2, 1)


        self.red = Ball('red')
        if pre_place_red:
            self.put_obj(self.red, self._rand_int(1, 3), 2)
        else:
            self.put_obj(self.red, 4, 1)

        self.blue = Ball('blue')
        if pre_place_blue:
            self.put_obj(self.blue, self._rand_int(1, 3), 2)
        else:
            self.put_obj(self.blue, 5, 1)

        if self.sticky:
            self.red.can_pickup = types.MethodType(lambda self: self.cur_pos[0] > 2, self.red)
            self.blue.can_pickup = types.MethodType(lambda self: self.cur_pos[0] > 2, self.blue)

        # Place the agent
        self.agent_pos = (self._rand_int(3, width-2), 2)
        self.agent_dir = 2

        self.red_goal = lambda: (0 <= self.red.cur_pos[0] <= 2)
        self.blue_goal = lambda: (0 <= self.blue.cur_pos[0] <= 2)

        if goal == 'red':
            self.goal_func = self.red_goal
        elif goal == 'blue':
            self.goal_func = self.blue_goal
        elif goal == 'both':
            self.goal_func = lambda: self.red_goal() and self.blue_goal()

    def _subtask_completions(self):
        completion = {}
        if self.compose:
            completion[self._gen_mission('both')] = self.red_goal() and self.blue_goal()
        else:
            completion[self._gen_mission('red')] = self.red_goal()
            completion[self._gen_mission('blue')] = self.blue_goal()
        return completion

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.goal_func():
            reward = self._reward()
            terminated = True
        return obs, reward, terminated, truncated, info
