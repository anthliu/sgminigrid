from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)
from minigrid.core.world_object import WorldObj

if TYPE_CHECKING:
    from minigrid.minigrid_env import MiniGridEnv

Point = Tuple[int, int]

# Map of object type to integers
NEW_OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
    "button": 11,
    "buttondoor": 12,
    "collectible": 13,
    "interactable": 14,
}
NEW_IDX_TO_OBJECT = dict(zip(NEW_OBJECT_TO_IDX.values(), NEW_OBJECT_TO_IDX.keys()))

class SGWorldObj(WorldObj):
    def __init__(self, type: str, color: str):
        assert type in NEW_OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos: Point | None = None

        # Current position of the object
        self.cur_pos: Point | None = None

    def encode(self) -> tuple[int, int, int]:
        """Encode the a description of this object as a 3-tuple of integers"""
        return (NEW_OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx: int, color_idx: int, state: int) -> WorldObj | None:
        """Create an object from a 3-tuple state description"""

        if type_idx in IDX_TO_OBJECT:
            return super().decode(type_idx, color_idx, state)

        obj_type = NEW_IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'button':
            v = Button(color, state == 1)
        elif obj_type == 'buttondoor':
            raise NotImplementedError
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

class Button(SGWorldObj):
    def __init__(self, color: str, is_pressed: bool = False):
        super().__init__("button", color)
        self.is_pressed = is_pressed

    def can_overlap(self):
        return True

    def see_behind(self):
        return True

    def toggle(self, env, pos):
        self.is_pressed = not self.is_pressed
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_pressed:
            state = 0
        else:
            state = 1

        return (NEW_OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]

        fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
        fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
        fill_coords(img, point_in_circle(0.5, 0.5, 0.4), c)

        if self.is_pressed:
            fill_coords(img, point_in_circle(0.5, 0.5, 0.35), (0, 0, 0))

class ButtonDoor(SGWorldObj):
    def __init__(self, button: SGWorldObj, color: str, is_open: bool = False):
        super().__init__("buttondoor", color)
        self.button = button
        self.is_open = is_open

    def can_overlap(self):
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        if self.button.is_pressed:
            self.is_open = not self.is_open
            return True
        else:
            return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        else:
            state = 1

        return (NEW_OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]
        button_c = COLORS[self.button.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            fill_coords(img, point_in_circle(cx=0.50, cy=0.50, r=0.30), 0.6 * np.array(button_c))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)

class Collectible(SGWorldObj):
    def __init__(self, name: str, color: str, env_state: dict, grid, obj_id=0):
        super().__init__("collectible", color)
        self.name = name
        self.env_state = env_state
        self.env_state[name] = False
        self.obj_id = obj_id
        self.grid = grid

    def can_overlap(self):
        return True

    def see_behind(self):
        return True

    def toggle(self, env, pos):
        self.grid.set(self.cur_pos[0], self.cur_pos[1], None)
        self.cur_pos = np.array([-1, -1])
        self.env_state[self.name] = True
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        return (NEW_OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.obj_id)

    def render(self, img):
        # temp rendering code
        c = COLORS[self.color]

        fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
        fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
        fill_coords(img, point_in_circle(0.5, 0.5, 0.4), c)

class Interactable(SGWorldObj):
    def __init__(self, name: str, color: str, env_state: dict, obj_id=0):
        super().__init__("interactable", color)
        self.name = name
        self.env_state = env_state
        self.env_state[name] = False
        self.obj_id = obj_id

    def can_overlap(self):
        return True

    def see_behind(self):
        return True

    def toggle(self, env, pos):
        self.env_state[self.name] = True
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        return (NEW_OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.obj_id)

    def render(self, img):
        # temp rendering code
        c = COLORS[self.color]

        fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
        fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
        fill_coords(img, point_in_circle(0.5, 0.5, 0.4), c)
        fill_coords(img, point_in_circle(0.5, 0.5, 0.35), (0, 0, 0))
