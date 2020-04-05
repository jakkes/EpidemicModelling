import torch
from epidemic import Model, ModelParameters
from epidemic.parse import parse_pop_swe

def run(params: ModelParameters):
    model = Model(params)
    
    s0 = params.population_grid
    e0 = torch.ones(params.grid.shape[0], params.no_age_groups, 1)
    i0 = torch.zeros(params.grid.shape[0], params.no_age_groups, 1)
    r0 = torch.zeros(params.grid.shape[0], params.no_age_groups, 1)

    model.set_start(s0, e0, i0, r0)

    for _ in range(10):
        model.prediction_step()

    print(model.S[10, 0, 0])
    print(model.E[10, 0, 0])
    print(model.I[10, 0, 0])
    print(model.R[10, 0, 0])

if __name__ == "__main__":
    
    regions = ['Blekinge' 'Dalarnas' 'Gotlands' 'Gävleborgs' 'Hallands' 'Jämtlands'
                'Jönköpings' 'Kalmar' 'Kronobergs' 'Norrbottens' 'Skåne' 'Stockholms'
                'Södermanlands' 'Uppsala' 'Värmlands' 'Västerbottens' 'Västernorrlands'
                'Västmanlands' 'Västra Götalands' 'Örebro' 'Östergötlands']

    # Each grid cell has a longitude and latitude
    grid = torch.zeros(21, 2)
    
    age_groups = [(0,9), (10,19), (20,29), (30,39), (40,49), (50,59), (60,69),(70,79),(80,100)]
    population_grid = torch.tensor(parse_pop_swe("./data/sweden-regional/population2019.csv", age_groups)).unsqueeze(2)
    
    # Age groups mix uniformly
    A = 0.5 * torch.ones(grid.shape[0], len(age_groups), len(age_groups))
    
    # Only travel within region
    G = 0.5 * torch.eye(grid.shape[0])

    env = ModelParameters(
        time_horizon=120,
        steps_per_day=10,
        age_groups=age_groups,
        gamma=0.15,
        R0=2.0,
        sigma=0.3,
        gridsize=grid.shape[0],
        A=A,
        G=G,
        population_grid=population_grid
    )

    run(env)