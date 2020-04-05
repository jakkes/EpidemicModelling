from typing import List

import torch
from torch import Tensor


class ModelParameters:
    def __init__(self, 
            time_horizon: int=None, 
            steps_per_day: int=None, 
            age_groups: List[str]=None,
            gamma: float=None,
            R0: float=None,
            sigma: float=None,
            gridsize: int=None,
            A: Tensor=None,
            G: Tensor=None,
            population_grid: Tensor=None
        ):
        """
        :param A: tensor of shape [grid_size, no_age_groups, no_age_groups]
        :param G: tensor of shape [grid_size, grid_size]
        """

        self.time_horizon: int = time_horizon
        self.steps_per_day: int = steps_per_day
        self.time_step: float = 1.0 / steps_per_day
        self.age_groups: List[str] = age_groups
        self.no_age_groups: int = len(age_groups)

        self.gamma: float = gamma
        self.R0: float = R0
        self.sigma: float = sigma

        self.gridsize: int = gridsize
        self.population_grid: Tensor = population_grid
        self.A: Tensor = A
        self.G: Tensor = G


class Model:
    def __init__(self, params: ModelParameters):
        
        self.params = params

        self.I = torch.zeros(params.time_horizon + 1, params.gridsize, params.no_age_groups, 1)
        self.S = torch.zeros(params.time_horizon + 1, params.gridsize, params.no_age_groups, 1) # change this
        self.E = torch.zeros(params.time_horizon + 1, params.gridsize, params.no_age_groups, 1)
        self.R = torch.zeros(params.time_horizon + 1, params.gridsize, params.no_age_groups, 1)

        self.t = -1

    def set_start(self, s: Tensor, e: Tensor, i: Tensor, r: Tensor):
        self.S[0] = s; self.I[0] = i; self.E[0] = e; self.R[0] = r
        self.t = 0

    def prediction_step(self):
        """Forward Euler"""
        if self.t < 0:
            raise ValueError("Must set initial data")

        self.t += 1

        self.S[self.t] = self.S[self.t-1]
        self.I[self.t] = self.I[self.t-1]
        self.E[self.t] = self.E[self.t-1]
        self.R[self.t] = self.R[self.t-1]

        dt = self.params.time_step
        for _ in range(self.params.steps_per_day):

            # G shape [gridsize, gridsize]
            # A shape [gridsize, age_groups, age_groups]
            # I shape [gridsize, age_groups, 1]

            # A.matmul(I) shape [1, gridsize, age_groups, 1]
            # G.unsqueeze.... shape [gridsize, gridsize, 1, 1]
            # output [gridsize, gridsize, agegroups, 1]

            s1 = self.S[self.t] * torch.sum(self.params.G.unsqueeze(2).unsqueeze(3) * self.params.A.matmul(self.I[self.t]).unsqueeze(0), dim=1)
            s2 = self.I[self.t] * torch.sum(self.params.G.unsqueeze(2).unsqueeze(3) * self.params.A.matmul(self.S[self.t]).unsqueeze(0), dim=1)

            # Internal X change
            isc = - self.params.gamma * self.params.R0 / self.params.population_grid * (s1 + s2)
            iec = - self.params.sigma * self.E[self.t]
            iic = - self.params.gamma * self.I[self.t]

            self.S[self.t] += dt * isc
            self.E[self.t] += dt * (iec - isc)
            self.I[self.t] += dt * (iic - iec)
            self.R[self.t] += dt * (-iic)