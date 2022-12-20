from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from sgminigrid.sgminigrid_env import SGMiniGridEnv
from sgminigrid.sgworld_object import Button, ButtonDoor


class ButtonDoorEnv(SGMiniGridEnv):

    def __init__(self, size=8, num_extra_buttons=2, max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 10 * size**2
        self.num_extra_buttons = num_extra_buttons
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES]
        )
        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps, **kwargs
        )

    @staticmethod
    def _gen_mission(color: str):
        return f"press the {color} button"

    def _gen_grid(self, width, height):
        allcolors = self._rand_subset(COLOR_NAMES, 2 + self.num_extra_buttons)

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(3, width - 3)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        self.place_agent()

        # Create a button with door
        dbutton_color = allcolors.pop()
        if self.agent_pos[0] >= splitIdx:
            is_pressed = True
        else:
            is_pressed = self._rand_bool()
        button = Button(dbutton_color, is_pressed=is_pressed)
        self.place_obj(button, size=(splitIdx, height))

        # Create a door in the wall
        if self.agent_pos[0] >= splitIdx:
            is_open = True
        else:
            is_open = self._rand_bool()
        doorIdx = self._rand_int(1, width - 2)
        self.put_obj(ButtonDoor(button, dbutton_color, is_open), splitIdx, doorIdx)

        # Create goal button
        goal_button_color = allcolors.pop()
        self.goal_button = Button(goal_button_color, is_pressed=False)
        self.place_obj(obj=self.goal_button, top=(splitIdx, 0), size=(width - splitIdx, height))
        self.mission = self._gen_mission(goal_button_color)

        # Create extra buttons
        for _ in range(self.num_extra_buttons):
            c = allcolors.pop()
            self.place_obj(Button(c, is_pressed=self._rand_bool()))
        
        assert len(allcolors) == 0

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.goal_button.is_pressed:
            reward = self._reward()
            terminated = True
        return obs, reward, terminated, truncated, info

