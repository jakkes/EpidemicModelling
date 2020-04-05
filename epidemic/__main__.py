from .env import Environment
from .model import Model

def run(env):
    pass

if __name__ == "__main__":
    
    age_groups = ["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80+"]

    env = Environment(
        time_horizon=120,
        steps_per_day=2,
        age_groups=age_groups
    )

    run(env)
