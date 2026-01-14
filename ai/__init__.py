from ai.models.actor_critic import DynamicActorCritic
from ai.algorithms import get_algorithm

def build_learner(config, state_dim, action_dims, backbone_name, algorithm_name):
    """
    Builds and returns the (model, learner) tuple.
    
    Args:
        config: Configuration object containing training settings.
        state_dim: Dimension of the input state.
        action_dims: List of action dimensions [Target, Role].
        backbone_name: Name of the backbone ('mlp', 'lstm', 'gru').
        algorithm_name: Name of the algorithm ('ppo', 'reinforce').
    
    Returns:
        model: The instantiated DynamicActorCritic model.
        learner: The instantiated algorithm learner (PPO or REINFORCE).
    """
    # Determine if the backbone implies recurrent processing
    is_recurrent = backbone_name.lower() in ["lstm", "gru"]
    
    # 1. Build Model
    # Using default hidden_dim=128, num_layers=2 as per class definition
    # If config has these values, they could be used here.
    model = DynamicActorCritic(
        state_dim=state_dim,
        action_dims=action_dims,
        backbone=backbone_name,
        hidden_dim=128,
        num_layers=2
    )

    # 2. Build Algorithm (Learner)
    learner = get_algorithm(algorithm_name, model, config, is_recurrent)
    
    return model, learner
