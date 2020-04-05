import torch

from epidemic import Model, ModelParameters

def run(params: ModelParameters):
    model = Model(params)

if __name__ == "__main__":
    

    age_groups = ["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80+"]
    
    grid = torch.zeros(20, 2)
    
    # Age groups mix uniformly
    A = 0.5 * torch.ones(grid.shape[0], len(age_groups), len(age_groups))
    
    # Only travel within region
    G = 0.5 * torch.eye(grid.shape[0])

    env = ModelParameters(
        time_horizon=120,
        steps_per_day=2,
        age_groups=age_groups,
        gamma=0.15,
        R0=2.0,
        sigma=0.3,
        grid=grid,
        A=A,
        G=G
    )

    run(env)