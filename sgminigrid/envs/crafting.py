from __future__ import annotations
import types

from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Ball, Wall
from sgminigrid.sgminigrid_env import SGMiniGridEnv
from sgminigrid.sgworld_object import Button, ButtonDoor, Collectible, Interactable
import numpy as np

LL_TASKS = ['wood', 'grass', 'iron', 'toolshed', 'workbench', 'factory']
HL_TASKS = ['plank', 'stick', 'cloth', 'rope', 'bridge', 'bed', 'axe', 'shears']
LL_TO_ID = {task: i for i, task in enumerate(LL_TASKS)}

VERB_TASK = {
    'get': ['wood', 'grass', 'iron'],
    'use': ['toolshed', 'workbench', 'factory'],
    'make': HL_TASKS
}
TASK_TO_VERB = {}
for verb, tasks in VERB_TASK.items():
    for t in tasks:
        TASK_TO_VERB[t] = verb

MAX_SUBGOAL = 4
SUBGOALS = {ll: [] for ll in LL_TASKS}
SUBGOALS['plank'] = ['wood', 'toolshed']
SUBGOALS['stick'] = ['wood', 'workbench']
SUBGOALS['cloth'] = ['grass', 'factory']
SUBGOALS['rope'] = ['grass', 'toolshed']
SUBGOALS['bridge'] = ['iron', 'wood', 'factory']
SUBGOALS['bed'] = ['wood', 'toolshed', 'grass', 'workbench']
SUBGOALS['axe'] = ['wood', 'workbench', 'iron', 'toolshed']
SUBGOALS['shears'] = ['wood', 'workbench', 'iron', 'workbench']

SUBGOAL_BASE_REWARD = 0.1
SUBGOAL_REWARDS = {ll: [] for ll in LL_TASKS}
SUBGOAL_REWARDS['plank'] = ['wood']
SUBGOAL_REWARDS['stick'] = ['wood']
SUBGOAL_REWARDS['cloth'] = ['grass']
SUBGOAL_REWARDS['rope'] = ['grass']
SUBGOAL_REWARDS['bridge'] = ['iron', 'wood']
SUBGOAL_REWARDS['bed'] = ['wood', 'plank', 'grass']
SUBGOAL_REWARDS['axe'] = ['wood', 'stick', 'iron']
SUBGOAL_REWARDS['shears'] = ['wood', 'stick', 'iron']

class Crafting(SGMiniGridEnv):
    def __init__(
        self,
        size=10,
        max_steps: int | None = None,
        compose=False,
        dist_bonus=False,
        **kwargs,
    ):
        self.compose = compose
        self.dist_bonus = dist_bonus
        if compose:
            self.place_holders = [LL_TASKS + HL_TASKS]
        else:
            self.place_holders = [LL_TASKS]
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=self.place_holders
        )
        completion_space = mission_space

        self.num_trees = 3
        self.num_grass = 5
        self.num_iron = 2

        if max_steps is None:
            max_steps = 10 * size

        # constants for calculating distance bonus
        self.size = size
        discount_factor = 0.95
        self.dist_bonus_scale = (1 - discount_factor) / 2
        self.pos_grid = np.mgrid[:size, :size].transpose(1, 2, 0)
        self.max_dist = 2 * self.size

        super().__init__(
            mission_space=mission_space,
            completion_space=completion_space,
            width=size,
            height=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        # override observation space with sketch
        sg_observation_space = spaces.Dict({
            'image': self.observation_space['image'],
            "direction": self.observation_space['direction'],
            "mission": self.observation_space['mission'],
            "mission_id": spaces.Discrete(self.mission_lookup.n_missions),
            "sketch": spaces.MultiDiscrete([len(LL_TASKS) + 1 for _ in range(MAX_SUBGOAL)]),
            "completion": spaces.MultiBinary(self.mission_lookup.n_missions)
        })
        self.observation_space = sg_observation_space

    @staticmethod
    def _gen_mission(target: str):
        # low level targets
        if target in LL_TASKS:
            return TASK_TO_VERB[target] + ' ' + target
        else:# high level targets
            return TASK_TO_VERB[target] + ' ' + target + ': ' + ', '.join(TASK_TO_VERB[sub] + ' ' + sub for sub in SUBGOALS[target])

    def _gen_grid(self, width, height):
        self.task_infos = {}# Info about current task for logging
        self.task_infos['tags'] = []

        # Sample goal
        if self.compose:
            self.goal = self._rand_elem(HL_TASKS)
        else:
            self.goal = self._rand_elem(LL_TASKS)
        self.task_infos['tags'].append(self.goal)
        self.mission = self._gen_mission(self.goal)
        self.sketch = np.zeros(MAX_SUBGOAL, dtype=np.int_)
        for i, sg in enumerate(SUBGOALS[self.goal]):
            self.sketch[i] = LL_TO_ID[sg] + 1

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.env_state = {}
        for t in LL_TASKS + HL_TASKS:
            self.env_state[t] = False
        self.prev_state = dict(self.env_state)

        # Generate objects and tools
        for _ in range(self.num_trees):
            self.tree = Collectible('wood', 'red', self.env_state, self.grid, 0)
            self.place_obj(self.tree)
        for _ in range(self.num_grass):
            self.grass = Collectible('grass', 'green', self.env_state, self.grid, 1)
            self.place_obj(self.grass)
        for _ in range(self.num_iron):
            self.iron = Collectible('iron', 'blue', self.env_state, self.grid, 2)
            self.place_obj(self.iron)

        self.toolshed = Interactable('toolshed', 'purple', self.env_state, 3)
        self.place_obj(self.toolshed)
        self.workbench = Interactable('workbench', 'yellow', self.env_state, 4)
        self.place_obj(self.workbench)
        self.factory = Interactable('factory', 'grey', self.env_state, 5)
        self.place_obj(self.factory)

        self.all_tools = ['toolshed', 'workbench', 'factory']
        self.target_to_obj = {
            'wood': self.tree,
            'grass': self.grass,
            'iron': self.iron,
            'toolshed': self.toolshed,
            'workbench': self.workbench,
            'factory': self.factory,
        }

        # Place the agent
        #self.agent_pos = (self._rand_int(3, width-2), 2)
        #self.agent_dir = 2
        self.place_agent()
        if self.dist_bonus:
            self.prev_pos = np.array(self.agent_pos)

    def _update_state(self):
        used_tool = {}
        for tool in self.all_tools:
            used_tool[tool] = not self.prev_state[tool] and self.env_state[tool]

        if used_tool['toolshed']:
            if self.env_state['wood']:
                self.env_state['plank'] = True
            if self.env_state['grass']:
                self.env_state['rope'] = True
            if self.env_state['stick'] and self.env_state['iron']:
                self.env_state['axe'] = True
        if used_tool['workbench']:
            if self.env_state['wood']:
                self.env_state['stick'] = True
            if self.env_state['plank'] and self.env_state['grass']:
                self.env_state['bed'] = True
            if self.env_state['stick'] and self.env_state['iron']:
                self.env_state['shears'] = True
        if used_tool['factory']:
            if self.env_state['grass']:
                self.env_state['cloth'] = True
            if self.env_state['iron'] and self.env_state['wood']:
                self.env_state['bridge'] = True

    def _subtask_completions(self):
        completion = {}
        if self.compose:
            tasks = HL_TASKS + LL_TASKS
        else:
            tasks = LL_TASKS
        for task in tasks:
            completion[self._gen_mission(task)] = self.env_state[task]
        return completion
    
    def _reward(self, obs):
        reward = 0
        subgoals = SUBGOAL_REWARDS[self.goal]
        obtained = {t for t in self.env_state if self.env_state[t] and not self.prev_state[t]}
        for sg in subgoals:
            if sg in obtained:
                reward += SUBGOAL_BASE_REWARD * (1.0 - 0.9 * (self.step_count / self.max_steps))

        if self.goal in obtained:
            if self.compose:
                reward += 1.0 - 0.9 * (self.step_count / self.max_steps)
            else:
                reward += 1.0
        if self.dist_bonus and not self.compose:
            target_enc = np.array(self.target_to_obj[self.goal].encode())
            pos = np.array(self.agent_pos)
            grid = self.grid.encode()
            grid[pos[0], pos[1]] = 0
            target_mask = (grid == target_enc).all(axis=-1)

            dist = np.abs(self.pos_grid - pos).sum(axis=-1)
            dist[~target_mask] = self.max_dist
            min_dist = dist.min()

            prev_dist = np.abs(self.pos_grid - self.prev_pos).sum(axis=-1)
            prev_dist[~target_mask] = self.max_dist
            prev_min_dist = prev_dist.min()

            bonus = self.dist_bonus_scale * max(0, prev_min_dist - min_dist)
            reward += bonus
            self.cur_min_dist = min_dist

            self.prev_pos = pos
        return reward

    def reset(self, *args, seed=None, options=None):
        obs, info = super().reset(*args, seed=seed, options=options)
        obs['sketch'] = self.sketch
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        obs['sketch'] = self.sketch

        self._update_state()
        reward = self._reward(obs)

        terminated = self.env_state[self.goal]

        # refresh tool usage
        for tool in self.all_tools:
            self.env_state[tool] = False
        self.prev_state = dict(self.env_state)

        return obs, reward, terminated, truncated, info
