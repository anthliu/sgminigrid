from minigrid.minigrid_env import MiniGridEnv

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
    ):
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
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "direction": spaces.Discrete(4),
                "mission": mission_space,
                "completion": None
            }
        )
    def _subtask_completions(self):
        return {}

    def reset(self, *args, seed=None, options=None):
        obs, info = super().reset(*args, seed=seed, options=options)
        obs['completion'] = self._subtask_completions()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs['completion'] = self._subtask_completions()
        return obs, reward, terminated, truncated, info
