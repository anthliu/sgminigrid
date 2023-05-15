#!/usr/bin/env python3

import time
import numpy as np
import gymnasium as gym
from sgminigrid.crafting_oracle import SOCraftingOracleAgent, CraftingOracleAgent, HLCraftingOracleAgent, SOHLCraftingOracleAgent
from sgminigrid.wrappers import CompactCraftObsWrapper

def shuffle(episodes, agent, env):
    actor = agent.get_test_actor()
    rs = []
    Ts = []
    for ep in range(episodes):
        r = 0
        T = 0
        obs, infos = env.reset()
        actor.observe_first(obs, infos)
        while True:
            T += 1
            # time.sleep(1/10.)
            # action = env.action_space.sample()
            # a = actor.act(sub_mission_id=obs['mission_id'], mission_id=5)
            a = actor.act()
            obs, reward, done, truncated, infos = env.step(a)
            r += reward
            terminal = done or truncated
            if terminal:
                break
            actor.observe(obs, a, infos)
        # print(f'Episode: reward {r:.2f}, time {T}')
        rs.append(r)
        Ts.append(T)
    rs = np.array(rs)
    Ts = np.array(Ts)
    print(f'Episode stats:')
    print(f'reward: {rs.mean():.2f} +/- {rs.std() / np.sqrt(rs.shape[0])}')
    print(f'time: {Ts.mean():.2f} +/- {Ts.std() / np.sqrt(Ts.shape[0])}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="SGMG-Crafting-Bonus-Fixed-v0"
    )
    parser.add_argument(
        "--episodes", default=1, type=int, help="n episodes"
    )
    parser.add_argument(
        "--render", action='store_true', help="render to screen"
    )
    parser.add_argument(
        "--second-order", action='store_true', help="use second order"
    )
    args = parser.parse_args()
    if args.render:
        env = CompactCraftObsWrapper(gym.make(args.env, render_mode='human'))
    else:
        env = CompactCraftObsWrapper(gym.make(args.env))

    if 'Compose' in args.env:
        if args.second_order:
            agent = SOHLCraftingOracleAgent(None, env, np.random.default_rng(42))
        else:
            agent = HLCraftingOracleAgent(None, env, np.random.default_rng(42))
    else:
        if args.second_order:
            agent = SOCraftingOracleAgent(None, env, np.random.default_rng(42))
        else:
            agent = CraftingOracleAgent(None, env, np.random.default_rng(42))

    shuffle(args.episodes, agent, env)
