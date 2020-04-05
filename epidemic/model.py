from typing import List

import torch
from torch import Tensor

class Model:
    def __init__(self, params: ModelParameters, grid: Tensor):
        
        self.params = params

        self.S = torch.zeros(params.time_horizon, params.no_age_groups, grid.shape[0], grid.shape[1])
        self.I = torch.zeros(params.time_horizon, params.no_age_groups, grid.shape[0], grid.shape[1])
        self.E = torch.zeros(params.time_horizon, params.no_age_groups, grid.shape[0], grid.shape[1])
        self.R = torch.zeros(params.time_horizon, params.no_age_groups, grid.shape[0], grid.shape[1])

        self.A = torch.zeros(params.no_age_groups, params.no_age_groups)
        


class ModelParameters:
    def __init__(self, 
            time_horizon: int=None, 
            steps_per_day: int=None, 
            age_groups: List[str]=None,
            
        ):
        self.time_horizon: int = time_horizon
        self.steps_per_day: int = steps_per_day
        self.time_step: float = 1.0 / steps_per_day
        self.age_groups: List[str] = age_groups
        self.no_age_groups: int = len(age_groups)