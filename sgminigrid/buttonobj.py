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

class Button(WorldObj):
    def __init__(self, color: str, is_pressed: bool = False):
        super().__init__("button", color)
        self.is_pressed = is_pressed

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_pressed

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

        #return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)
        return (11, COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_pressed:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)
