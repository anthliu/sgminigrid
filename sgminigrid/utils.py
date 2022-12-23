import itertools as it
import numpy as np

class MissionLookup(object):
    """
    Utility functions for environment mission spaces
    """
    def __init__(self, mission_space):
        self.mission_space = mission_space
        self.id_to_mission = []
        self.mission_to_id = {}
        mission_args = mission_space.ordered_placeholders
        if mission_space.ordered_placeholders is not None:
            mission_args = it.product(*mission_args)
        else:
            mission_args = [[]]
        for i, samples in enumerate(mission_args):
            mission = mission_space.mission_func(*samples)
            self.id_to_mission.append(mission)

            self.mission_to_id.setdefault(mission, i)

        self.n_missions = len(self.id_to_mission)

    def dict_to_vec(self, completion_dict):
        vec = np.zeros(self.n_missions, dtype=np.bool_)
        for subtask, val in completion_dict.items():
            vec[self.mission_to_id[subtask]] = val
        return vec
