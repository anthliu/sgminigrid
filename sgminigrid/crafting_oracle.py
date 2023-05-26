from copy import deepcopy
import numpy as np
import itertools as it

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

def _fill_flow(reachable_q, flow, border):
    while len(reachable_q) > 0:
        state = reachable_q.pop(0)
        dist = flow[state]
        for nstate, _ in _get_r_neighbors(*state, border):
            ndist = flow[nstate]
            if dist + 1 < ndist:
                flow[nstate] = dist + 1
                reachable_q.append(nstate)

class CraftingOracleAgent():
    TAGS = []
    def __init__(self, args, env, rng, nth_order=1):
        self.args = args
        self.rng = rng
        assert isinstance(env, CompactCraftObsWrapper)
        # assert self.observation_space['mission_id'].n == 6
        self.observation_space = deepcopy(env.observation_space)
        self.observation_space['mission_id'].n = 6# override environment to have 6 goals
        self.action_space = env.action_space
        self.actor = CraftingOracleActor(self.args, self.rng, self.observation_space, self.action_space, nth_order=nth_order)

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

class BaseLearner(object):
    def observe_first(self, obs, infos):
        pass

    def observe(self, obs, next_obs, action, reward, done, infos, next_infos):
        pass

    def update(self, accum=None):
        pass

class CraftingOracleActor(object):
    def __init__(self, args, rng, observation_space, action_space, nth_order=1):
        self.args = args
        self.rng = rng
        self.observation_space = observation_space
        self.action_space = action_space

        self.h, self.w = self.observation_space['image'].shape
        self.base_grid = np.zeros((self.h, self.w), np.uint8)
        self.flow = None
        self.border = None
        self.orders = nth_order

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
        self.flow = np.full((*([MISSIONS]*self.orders), self.h, self.w, 4), 3 * self.h * self.w + 4, dtype=np.int_)

        for order in range(self.orders):
            for missions in it.permutations(range(MISSIONS), order+1):
                reachable_q = []
                nth_order_mission = missions + (missions[-1],) * (self.orders - order - 1)
                prev_order_mission = missions[1:] + (missions[-1],) * (self.orders - order)
                assert len(nth_order_mission) == self.orders
                obj_id = MISSION_OBJ[missions[0]]
                x_goal, y_goal = np.where(self.base_grid == obj_id)
                cur_flow = self.flow[nth_order_mission]
                prev_flow = self.flow[prev_order_mission]
                for i in range(x_goal.shape[0]):
                    for (x, y, d), _ in _goal_neighbors(x_goal[i], y_goal[i], self.border):
                        if order == 0:
                            cur_flow[x, y, d] = 1
                        else:
                            cur_flow[x, y, d] = prev_flow[x, y, d] + 1
                        reachable_q.append((x, y, d))
                _fill_flow(reachable_q, cur_flow, self.border)

    def observe(self, next_obs, action, infos):
        self.last_obs = next_obs
        self.last_infos = infos

    def act(self, accum=None, missions=None, include_terminal=False):
        if missions is None:
            missions = (self.last_obs['mission_id'],) * self.orders
        assert len(missions) == self.orders

        pos = np.where(self.last_obs['image'] == AGENT_ID)
        x, y = pos[0][0], pos[1][0]
        d = self.last_obs['direction']
        #max_dist = self.flow[sub_mission_id, mission_id, x, y, d]
        cur_flow = self.flow[missions]
        max_dist = cur_flow[x, y, d]
        a = None
        for (nx, ny, nd), na in _get_neighbors(x, y, d, self.border):
            if cur_flow[nx, ny, nd] < max_dist:
                max_dist = cur_flow[nx, ny, nd]
                a = na
        if a is None:
            a = MiniGridEnv.Actions.toggle
        if include_terminal:
            return a, (a == MiniGridEnv.Actions.toggle)
        else:
            return a

class HLCraftingOracleAgent(CraftingOracleAgent):
    TAGS = []
    def __init__(self, args, env, rng, nth_order=1):
        self.args = args
        self.rng = rng
        # assert isinstance(env, CompactCraftObsWrapper)
        self.observation_space = env.observation_space
        assert self.observation_space['mission_id'].n == 8
        self.action_space = env.action_space
        self.actor = HLCraftingOracleActor(self.args, self.rng, self.observation_space, self.action_space, nth_order=nth_order)

class HLCraftingOracleActor(object):
    def __init__(self, args, rng, observation_space, action_space, nth_order=1):
        self.args = args
        self.rng = rng
        self.observation_space = observation_space
        self.action_space = action_space

        self.ll_actor = CraftingOracleActor(args, rng, observation_space, action_space, nth_order=nth_order)
        self.orders = nth_order

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
        if (sketch == [1, 5, 3, 5]).all():
            # shears plan can be shortened
            sketch = np.array([1, 3, 5])
        self.seen_completed = self.seen_completed | self.last_obs['completion']
        completion = self.seen_completed
        sketch = [m - 1 for m in sketch if m > 0]# remove padding
        plan = [m for m in sketch if not completion[m]]

        hl_action = tuple(plan[0:self.orders])
        hl_action = hl_action + (hl_action[-1],) * (self.orders - len(hl_action))
        a = self.ll_actor.act(missions=hl_action)
        return a
