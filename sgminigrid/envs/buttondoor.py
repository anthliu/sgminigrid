from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from sgminigrid.sgminigrid_env import SGMiniGridEnv
from sgminigrid.sgworld_object import Button, ButtonDoor

class ButtonDoorEnv(SGMiniGridEnv):

    def __init__(self, size=8, num_colors=6, num_extra_buttons=2, max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 10 * size**2
        self.num_extra_buttons = num_extra_buttons
        self.colors = COLOR_NAMES[:num_colors]
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[self.colors, ["button"]]
        )
        completion_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[self.colors, ["button", "buttondoor"]]
        )
        super().__init__(
            mission_space=mission_space, completion_space=completion_space,
            grid_size=size, max_steps=max_steps, **kwargs
        )

    @staticmethod
    def _gen_mission(color: str, obj_type: str='button'):
        if obj_type == 'button':
            return f"press the {color} button"
        elif obj_type == 'buttondoor':
            return f"open the {color} door"

    def _gen_grid(self, width, height):
        self.task_infos = {}# Info about current task for logging
        self.task_infos['tags'] = []
        allcolors = self._rand_subset(self.colors, 1 + self.num_extra_buttons)
        self.allbuttons = {}
        buttons_before_door = []
        buttons_after_door = []

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create a vertical splitting wall
        #splitIdx = self._rand_int(2, width - 2)
        splitIdx = width // 2
        self.grid.vert_wall(splitIdx, 0)

        task_difficulty = self._rand_elem(['open', 'openlong', 'closed', 'closedlocked'])
        self.task_infos['tags'].append(task_difficulty)

        if task_difficulty == 'open':
            agent_passed_door = self._rand_bool()
            goal_passed_door = agent_passed_door
            open_door = agent_passed_door
            door_button_pressed = open_door or self._rand_bool()
        elif task_difficulty == 'openlong':
            agent_passed_door = False
            goal_passed_door = True
            open_door = True
            door_button_pressed = True
        elif task_difficulty == 'closed':
            agent_passed_door = False
            goal_passed_door = True
            open_door = False
            door_button_pressed = True
        elif task_difficulty == 'closedlocked':
            agent_passed_door = False
            goal_passed_door = True
            open_door = False
            door_button_pressed = False
        else:
            raise ValueError

        # Create a button with door
        dbutton_color = allcolors.pop()
        button = Button(dbutton_color, is_pressed=door_button_pressed)
        self.place_obj(button, size=(splitIdx, height))
        buttons_before_door.append(button)
        self.allbuttons[dbutton_color] = button

        #doorIdx = self._rand_int(1, width - 2)
        doorIdx = height // 2
        self.door = ButtonDoor(button, dbutton_color, open_door)
        self.put_obj(self.door, splitIdx, doorIdx)

        # Create rest of the buttons
        top = (splitIdx, 1)
        for _ in range(self.num_extra_buttons):
            c = allcolors.pop()
            extra_button = Button(c, is_pressed=self._rand_bool())
            self.place_obj(extra_button, top=top)# place at least one button after the door
            top = None
            self.allbuttons[c] = extra_button
            if extra_button.cur_pos[0] >= splitIdx:
                buttons_after_door.append(extra_button)
            else:
                buttons_before_door.append(extra_button)

        assert len(allcolors) == 0
        assert len(buttons_after_door) >= 1
        assert len(buttons_before_door) >= 1

        # Select goal button
        if goal_passed_door:
            self.goal_button = self._rand_elem(buttons_after_door)
        else:
            self.goal_button = self._rand_elem(buttons_before_door)
        self.goal_button.is_pressed = False

        self.task_infos['tags'].append(self.goal_button.color)
        self.mission = self._gen_mission(self.goal_button.color)

        # Place the agent
        if agent_passed_door:
            self.place_agent(top=(splitIdx, 1))
        else:
            self.place_agent(size=(splitIdx, height))
        
    def _subtask_completions(self):
        completion = {}
        for c, button in self.allbuttons.items():
            completion[self._gen_mission(c)] = button.is_pressed
        completion[self._gen_mission(self.door.color, 'buttondoor')] = self.door.is_open
        return completion

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.goal_button.is_pressed:
            reward = self._reward()
            terminated = True
        return obs, reward, terminated, truncated, info

