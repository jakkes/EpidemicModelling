import pandas as pd
import torch
import numpy as np

from epidemic import Model, ModelParameters
from epidemic.parse import parse_pop_swe
from epidemic.coropleth import plot_sweden


def run_model(params: ModelParameters):
    model = Model(params)

    s0 = params.population_grid
    e0 = torch.ones(params.gridsize, params.no_age_groups, 1)
    i0 = torch.zeros(params.gridsize, params.no_age_groups, 1)
    r0 = torch.zeros(params.gridsize, params.no_age_groups, 1)

    model.set_start(s0, e0, i0, r0)

    for _ in range(params.time_horizon):
        model.prediction_step()

    return model


if __name__ == "__main__":

    regions = np.array([
            "Blekinge", "Dalarnas", "Gotlands", "Gävleborgs", "Hallands", "Jämtlands", 
            "Jönköpings", "Kalmar", "Kronobergs", "Norrbottens", "Skåne", "Stockholms", 
            "Södermanlands", "Uppsala", "Värmlands", "Västerbottens", "Västernorrlands", 
            "Västmanlands", "Västra Götalands", "Örebro", "Östergötlands"])

    age_groups = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 100)]
    population_grid = torch.tensor(
        parse_pop_swe("./data/sweden-regional/population2019.csv", age_groups)
    ).unsqueeze(2)

    # Age groups mix uniformly
    A = 0.5 * torch.ones(population_grid.shape[0], len(age_groups), len(age_groups))

    # Travel is done only within region
    G = 0.5 * torch.eye(population_grid.shape[0])

    env = ModelParameters(
        time_horizon=30,
        steps_per_day=10,
        age_groups=age_groups,
        gamma=0.15,
        R0=2.0,
        sigma=0.3,
        gridsize=len(regions),
        A=A,
        G=G,
        population_grid=population_grid,
    )

    model = run_model(env)

    S = model.S[-1].sum(1).view(-1)
    E = model.E[-1].sum(1).view(-1)
    I = model.I[-1].sum(1).view(-1)
    R = model.R[-1].sum(1).view(-1)

    data = torch.stack((S, E, I, R), dim=1).numpy()
    data = np.concatenate((data, regions.reshape(-1, 1)), axis=1)
    data = pd.DataFrame(data=data, columns=["S", "E", "I", "R", "region"])
    
    plot_sweden(data)