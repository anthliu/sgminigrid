from copy import deepcopy
import numpy as np

from minigrid.minigrid_env import MiniGridEnv
from sgminigrid.wrappers import CompactCraftObsWrapper

DIR_MAP = np.array([
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
], dtype=np.int_)

MISSION_OBJ = np.array([4, 5, 6, 7, 8, 9])
MISSIONS = 6
AGENT_ID = 3

def _get_neighbors(x, y, d, border):
    xd, yd = DIR_MAP[d]
    result = [((x, y, (d-1)%4), MiniGridEnv.Actions.left), ((x, y, ((d+1)%4)), MiniGridEnv.Actions.right)]
    if not border[x+xd, y+yd]:
        result.append(((x+xd, y+yd, d), MiniGridEnv.Actions.forward))
    return result

def _get_r_neighbors(x, y, d, border):
    xd, yd = DIR_MAP[d]
    result = [((x, y, (d-1)%4), MiniGridEnv.Actions.right), ((x, y, ((d+1)%4)), MiniGridEnv.Actions.left)]
    if not border[x-xd, y-yd]:
        result.append(((x-xd, y-yd, d), MiniGridEnv.Actions.forward))
    return result

def _goal_neighbors(x, y, border):
    return [
        ((x-DIR_MAP[d][0], y-DIR_MAP[d][1], d), MiniGridEnv.Actions.toggle)
        for d in range(4)
        if not border[x-DIR_MAP[d][0], y-DIR_MAP[d][1]]
    ]

class CraftingOracleAgent():
    TAGS = []
    def __init__(self, args, env, rng):
        self.args = args
        self.rng = rng
        assert isinstance(env, CompactCraftObsWrapper)
        # assert self.observation_space['mission_id'].n == 6
        self.observation_space = deepcopy(env.observation_space)
        self.observation_space['mission_id'].n = 6# override environment to have 6 goals
        self.action_space = env.action_space
        self.actor = CraftingOracleActor(self.args, self.rng, self.observation_space, self.action_space, second_order=False)

    def get_train_actor(self):
        return self.actor

    def get_test_actor(self):
        return self.actor

    def get_learner(self):
        return BaseLearner()

    def save(self, checkpoint_dir):
        pass

    def load(self, checkpoint_dir):
        pass

class SOCraftingOracleAgent(CraftingOracleAgent):
    TAGS = ['second_order']
    def __init__(self, args, rng, env):
        super().__init__(args, rng, env)
        self.actor = CraftingOracleActor(self.args, self.rng, self.observation_space, self.action_space, second_order=True)

class BaseLearner(object):
    def observe_first(self, obs, infos):
        pass

    def observe(self, obs, next_obs, action, reward, done, infos, next_infos):
        pass

    def update(self, accum=None):
        pass

class CraftingOracleActor(object):
    def __init__(self, args, rng, observation_space, action_space, second_order=False):
        self.args = args
        self.rng = rng
        self.observation_space = observation_space
        self.action_space = action_space

        self.h, self.w = self.observation_space['image'].shape
        self.base_grid = np.zeros((self.h, self.w), np.uint8)
        self.flow = None
        self.border = None
        self.second_order = second_order

    def observe_first(self, obs, infos):
        self.last_obs = obs
        self.last_infos = infos
        
        # flow map
        new_base_grid = obs['image'].copy()
        new_base_grid[new_base_grid == AGENT_ID] = 1# remove agent
        if (self.base_grid == new_base_grid).all():
            return

        # build flow map
        self.base_grid = new_base_grid
        self.border = self.base_grid == 2
        self.flow = np.full((MISSIONS, MISSIONS, self.h, self.w, 4), 3 * self.h * self.w + 4, dtype=np.int_)
        reachable_q = []
        for mission in range(MISSIONS):
            obj_id = MISSION_OBJ[mission]
            x_goal, y_goal = np.where(self.base_grid == obj_id)
            for i in range(x_goal.shape[0]):
                for (x, y, d), _ in _goal_neighbors(x_goal[i], y_goal[i], self.border):
                    self.flow[mission, mission, x, y, d] = 1
                    reachable_q.append((mission, mission, x, y, d))

        while len(reachable_q) > 0:
            sm, m, x, y, d = reachable_q.pop(0)
            dist = self.flow[sm, m, x, y, d]
            for (nx, ny, nd), _ in _get_r_neighbors(x, y, d, self.border):
                ndist = self.flow[sm, m, nx, ny, nd]
                if dist + 1 < ndist:
                    self.flow[sm, m, nx, ny, nd] = dist + 1
                    reachable_q.append((sm, m, nx, ny, nd))

        if self.second_order:
            for mission in range(MISSIONS):
                for sub_mission in range(MISSIONS):
                    if mission == sub_mission:
                        continue
                    obj_id = MISSION_OBJ[sub_mission]
                    x_goal, y_goal = np.where(self.base_grid == obj_id)
                    for i in range(x_goal.shape[0]):
                        for (x, y, d), _ in _goal_neighbors(x_goal[i], y_goal[i], self.border):
                            self.flow[sub_mission, mission, x, y, d] = self.flow[mission, mission, x, y, d] + 1
                            reachable_q.append((sub_mission, mission, x, y, d))

            while len(reachable_q) > 0:
                sm, m, x, y, d = reachable_q.pop(0)
                dist = self.flow[sm, m, x, y, d]
                for (nx, ny, nd), _ in _get_r_neighbors(x, y, d, self.border):
                    ndist = self.flow[sm, m, nx, ny, nd]
                    if dist + 1 < ndist:
                        self.flow[sm, m, nx, ny, nd] = dist + 1
                        reachable_q.append((sm, m, nx, ny, nd))

    def observe(self, next_obs, action, infos):
        self.last_obs = next_obs
        self.last_infos = infos

    def act(self, accum=None, mission_id=None, sub_mission_id=None):
        if mission_id is None:
            mission_id = self.last_obs['mission_id']
        if sub_mission_id is None:
            sub_mission_id = mission_id
        else:
            assert self.second_order

        pos = np.where(self.last_obs['image'] == AGENT_ID)
        x, y = pos[0][0], pos[1][0]
        d = self.last_obs['direction']
        max_dist = self.flow[sub_mission_id, mission_id, x, y, d]
        a = None
        for (nx, ny, nd), na in _get_neighbors(x, y, d, self.border):
            if self.flow[sub_mission_id, mission_id, nx, ny, nd] < max_dist:
                max_dist = self.flow[sub_mission_id, mission_id, nx, ny, nd]
                a = na
        if a is None:
            return MiniGridEnv.Actions.toggle
        return a

class HLCraftingOracleAgent(CraftingOracleAgent):
    TAGS = []
    def __init__(self, args, env, rng):
        self.args = args
        self.rng = rng
        # assert isinstance(env, CompactCraftObsWrapper)
        self.observation_space = env.observation_space
        assert self.observation_space['mission_id'].n == 8
        self.action_space = env.action_space
        self.actor = HLCraftingOracleActor(self.args, self.rng, self.observation_space, self.action_space, second_order=False)

class SOHLCraftingOracleAgent(HLCraftingOracleAgent):
    def __init__(self, args, rng, env):
        super().__init__(args, rng, env)
        self.actor = HLCraftingOracleActor(self.args, self.rng, self.observation_space, self.action_space, second_order=True)

class HLCraftingOracleActor(object):
    def __init__(self, args, rng, observation_space, action_space, second_order=False):
        self.args = args
        self.rng = rng
        self.observation_space = observation_space
        self.action_space = action_space

        self.second_order = second_order
        self.ll_actor = CraftingOracleActor(args, rng, observation_space, action_space, second_order)

    def observe_first(self, obs, infos):
        self.last_obs = obs
        self.last_infos = infos
        self.ll_actor.observe_first(obs, infos)

        self.seen_completed = np.zeros(self.observation_space['completion'].shape, np.bool_)

    def observe(self, next_obs, action, infos):
        self.last_obs = next_obs
        self.last_infos = infos
        self.ll_actor.observe(next_obs, action, infos)

    def act(self, accum=None, mission_id=None):
        if mission_id is None:
            mission_id = self.last_obs['mission_id']

        sketch = self.last_obs['sketch']
        self.seen_completed = self.seen_completed | self.last_obs['completion']
        completion = self.seen_completed
        sketch = [m - 1 for m in sketch if m > 0]# remove padding
        plan = [m for m in sketch if not completion[m]]
        if len(plan) == 0:
            plan = [sketch[-1]]# for shears edge case (workbench twice)
        if not self.second_order or len(plan) < 2:
            hl_action = plan[0]
            a = self.ll_actor.act(mission_id=hl_action)
        else:
            hl_action = plan[0:2]
            a = self.ll_actor.act(mission_id=hl_action[1], sub_mission_id=hl_action[0])
        return a
