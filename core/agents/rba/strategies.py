"""
Decision-making strategies for the Rule-Based Agent.
"""
import numpy as np
from typing import TYPE_CHECKING
from .constants import RBAEventType

if TYPE_CHECKING:
    from .engine import BayesianInferenceEngine

# Base values, formerly LIKELIHOOD_MATRIX
BASE_LIKELIHOODS = {
    RBAEventType.PLAYER_VOTED: 1.1,
    RBAEventType.PLAYER_WAS_VOTED_FOR: 1.0,
    RBAEventType.PLAYER_ACCUSED: 1.5,
    RBAEventType.PLAYER_DEFENDED: 0.7,
    RBAEventType.HEAL_ATTEMPT: 0.5,
    RBAEventType.INVESTIGATE_ATTEMPT: 0.5,
    RBAEventType.ACCUSATION_WAS_WRONG: 2.0,
    RBAEventType.ACCUSATION_WAS_RIGHT: 0.1,
}

def get_contextual_likelihood(event_type: RBAEventType, day: int, survivor_count: int, actor_id: int, engine: 'BayesianInferenceEngine') -> float:
    """
    Calculates event likelihood P(E|M) based on game context, including correlations.
    """
    likelihood = BASE_LIKELIHOODS.get(event_type, 1.0)

    # --- Survivor Count Adjustments (End-game logic) ---
    if survivor_count <= 4:
        if event_type == RBAEventType.PLAYER_ACCUSED: likelihood = 1.2
        elif event_type == RBAEventType.PLAYER_DEFENDED: likelihood = 0.85

    # --- Day-based Adjustments ---
    if day >= 3:
        if event_type == RBAEventType.PLAYER_VOTED: likelihood = 1.2
        elif event_type == RBAEventType.ACCUSATION_WAS_WRONG: likelihood = 2.5

    # --- Correlation-based Adjustments ---
    if event_type in [RBAEventType.PLAYER_VOTED, RBAEventType.PLAYER_ACCUSED]:
        correlation_vector = engine.correlation[actor_id]
        mafia_probabilities = engine.posterior
        
        # Calculate the average suspicion of players the actor is correlated with
        if np.sum(correlation_vector) > 0:
            correlated_suspicion = np.dot(correlation_vector, mafia_probabilities) / np.sum(correlation_vector)
        else:
            correlated_suspicion = 0.0

        # Adjust likelihood based on the average suspicion of correlated players.
        # A 'neutral' suspicion level is ~0.3. If friends are more suspicious, this action is more suspicious.
        adjustment = (correlated_suspicion - 0.3) * 0.5 # The 0.5 dampens the effect
        likelihood += adjustment

    return likelihood

def calculate_entropy(probabilities):
    """
    Calculates the entropy of a probability distribution.
    H(P) = -sum(p_i * log2(p_i))
    """
    probabilities = np.where(probabilities == 0, 1e-9, probabilities)
    return -np.sum(probabilities * np.log2(probabilities))

def should_abstain(entropy, threshold):
    """
    Decides whether to abstain based on entropy.
    """
    return entropy > threshold

def softmax_selection(probabilities, temperature=1.0):
    """
    Selects an action based on softmax sampling of probabilities.
    """
    probabilities = np.asarray(probabilities).astype('float64')
    # Use log-softmax for numerical stability
    log_probs = np.log(probabilities + 1e-9) / temperature
    exp_probs = np.exp(log_probs - np.max(log_probs))
    softmax_probs = exp_probs / np.sum(exp_probs)
    
    if np.isnan(softmax_probs).any():
        return np.random.choice(len(probabilities))

    return np.random.choice(len(softmax_probs), p=softmax_probs)
