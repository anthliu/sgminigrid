import numpy as np
from gymnasium.core import ObservationWrapper, ObsType, Wrapper
from gymnasium import spaces
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

class CompactCraftObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=11,
            shape=(self.env.width, self.env.height),  # number of cells
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )

        return {**obs, "image": self._compress(full_grid)}

    @staticmethod
    def _compress(obs):
        buff = obs[:, :, 0].copy()

        agent = obs[:, :, 0] == 10
        buff[agent] = 3

        collectibles = obs[:, :, 0] == 13
        buff[collectibles] = 4 + obs[collectibles, 2]

        interactables = obs[:, :, 0] == 14
        buff[interactables] = 4 + obs[interactables, 2]
        return buff
