from minigrid.minigrid_env import MiniGridEnv

class SGMiniGridEnv(MiniGridEnv):
    def train(self):
        pass
    def eval(self):
        pass

    def _subtask_completions(self):
        return {}

    def reset(self, *args, seed=None, options=None):
        self.task_infos = {}
        obs, info = super().reset(*args, seed=seed, options=options)
        info['completion'] = self._subtask_completions()
        info.update(self.task_infos)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info['completion'] = self._subtask_completions()
        info.update(self.task_infos)
        return obs, reward, terminated, truncated, info
