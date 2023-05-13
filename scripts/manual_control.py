#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym

from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.window import Window
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from sgminigrid.wrappers import CompactCraftObsWrapper


class ManualControl:
    def __init__(
        self,
        env: MiniGridEnv,
        agent_view: bool = False,
        window: Window = None,
        task_id=None,
        seed=None,
    ) -> None:
        self.env = env
        self.agent_view = agent_view
        self.seed = seed
        self.task_id = task_id

        if window is None:
            window = Window("minigrid - " + str(env.__class__))
        self.window = window
        self.window.reg_key_handler(self.key_handler)

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)
        self.window.show(block=True)

    def step(self, action: MiniGridEnv.Actions):
        obs, reward, terminated, truncated, info = self.env.step(action)
        print(obs)
        print(f"mission_id={obs['mission_id']} step={self.env.step_count}, reward={reward:.2f} completion={obs['completion']}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.redraw()

    def redraw(self):
        frame = self.env.get_frame(agent_pov=self.agent_view)
        self.window.show_img(frame)

    def reset(self, seed=None):
        self.env.reset(seed=seed, options={'task_id': self.task_id})

        if hasattr(self.env, "mission"):
            #print("Mission: %s" % self.env.mission)
            print(f"Mission: {self.env.mission}, ID: {self.env.mission_lookup.mission_to_id[self.env.mission]}")
            self.window.set_caption(self.env.mission)

        self.redraw()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.window.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": MiniGridEnv.Actions.left,
            "right": MiniGridEnv.Actions.right,
            "up": MiniGridEnv.Actions.forward,
            " ": MiniGridEnv.Actions.toggle,
            "pageup": MiniGridEnv.Actions.pickup,
            "pagedown": MiniGridEnv.Actions.drop,
            "enter": MiniGridEnv.Actions.done,
        }

        action = key_to_action[key]
        self.step(action)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="SGMG-BDoor-v0"
    )
    parser.add_argument(
        "--task-id", type=int, help="Task to load in the environment", default=None
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )
    parser.add_argument(
        "--eval",
        default=False,
        help="Eval env version",
        action="store_true",
    )

    args = parser.parse_args()

    env: MiniGridEnv = gym.make(args.env, tile_size=args.tile_size)
    env = CompactCraftObsWrapper(env)

    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, env.tile_size)
        env = ImgObsWrapper(env)

    if args.eval:
        env.eval()
    else:
        env.train()
    manual_control = ManualControl(env, agent_view=args.agent_view, seed=args.seed, task_id=args.task_id)
    manual_control.start()
