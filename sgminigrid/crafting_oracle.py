from sgminigrid.envs.crafting import Crafting

class CraftingOracleAgent():
    TAGS = ['second_order']
    def __init__(self, args, rng, env):
        self.args = args
        self.rng = rng
        assert isinstance(env, Crafting)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def get_train_actor(self):
        return CraftingOracleActor(self.args, self.rng, self.observation_space)

    def get_test_actor(self):
        return CraftingOracleActor(self.args, self.rng, self.observation_space)

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
    def __init__(self, args, rng, observation_space):
        self.args = args
        self.rng = rng
        self.observation_space = observation_space

    def observe_first(self, obs, infos):
        self.last_obs = obs
        self.last_infos = infos

    def observe(self, next_obs, action, infos):
        self.last_obs = next_obs
        self.last_infos = infos

    def act(self, accum=None, mission_id=None, sub_mission_id=None):
        if mission_id is None:
            mission_id = self.last_obs['mission_id']

        pass
