from typing import List

class Environment:
    def __init__(self, time_horizon: int=None, steps_per_day: int=None, age_groups: List[str]=None):
        self.time_horizon: int = time_horizon
        self.steps_per_day: int = steps_per_day
        self.time_step: float = 1.0 / steps_per_day
        self.age_groups: List[str] = age_groups
        self.no_age_groups: int = len(age_groups)