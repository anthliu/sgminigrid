from gymnasium import spaces
from minigrid.core.constants import TILE_PIXELS
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from sgminigrid.utils import MissionLookup

class SGMiniGridEnv(MiniGridEnv):
    def __init__(
        self,
        mission_space: MissionSpace,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: str | None = None,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
        completion_space: MissionSpace | None = None,
    ):
        #self.completion_space = mission_space
        super().__init__(
            mission_space,
            grid_size,
            width,
            height,
            max_steps,
            see_through_walls,
            agent_view_size,
            render_mode,
            highlight,
            tile_size,
            agent_pov,
        )
        if completion_space is None:
            self.completion_space = mission_space
        else:
            self.completion_space = completion_space
        self.mission_lookup = MissionLookup(self.completion_space)
        sg_observation_space = spaces.Dict({
            'image': self.observation_space['image'],
            "direction": self.observation_space['direction'],
            "mission": self.observation_space['mission'],
            "mission_id": spaces.Discrete(self.mission_lookup.n_missions),
            "completion": spaces.MultiBinary(self.mission_lookup.n_missions)
        })
        self.observation_space = sg_observation_space

    def train(self):
        pass
    def eval(self):
        pass

    def _subtask_completions(self):
        return {}

    def _reward(self):
        return 1.0

    def reset(self, *args, seed=None, options=None):
        self.task_infos = {}
        obs, info = super().reset(*args, seed=seed, options=options)
        obs['completion'] = self.mission_lookup.dict_to_vec(self._subtask_completions())
        obs['mission_id'] = self.mission_lookup.mission_to_id[obs['mission']]
        info.update(self.task_infos)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs['completion'] = self.mission_lookup.dict_to_vec(self._subtask_completions())
        obs['mission_id'] = self.mission_lookup.mission_to_id[obs['mission']]
        info.update(self.task_infos)
        return obs, reward, terminated, truncated, info
