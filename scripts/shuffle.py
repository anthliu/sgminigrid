#!/usr/bin/env python3

import numpy as np
import gymnasium as gym

def shuffle(env):
    env.reset()
    T = 0
    while True:
        T += 1
        action = env.action_space.sample()
        observation, reward, done, truncated, infos = env.step(action)
        terminal = done or truncated
        if terminal:
            break
    print(f'T = {T}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="SGMG-BDoor-v0"
    )
    args = parser.parse_args()
    env = gym.make(args.env, render_mode='human')

    shuffle(env)
