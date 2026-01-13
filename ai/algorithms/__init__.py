from .ppo import PPO
from .reinforce import REINFORCE

def get_algorithm(name, model, config, is_recurrent=False):
    name = name.lower()
    if name == "ppo":
        return PPO(model, config, is_recurrent)
    elif name == "reinforce":
        return REINFORCE(model, config, is_recurrent)
    else:
        raise ValueError(f"Unknown algorithm: {name}")
