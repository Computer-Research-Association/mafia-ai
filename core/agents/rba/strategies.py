"""
Decision-making strategies for the Rule-Based Agent.
"""
import numpy as np

def calculate_entropy(probabilities):
    """
    Calculates the entropy of a probability distribution.
    H(P) = -sum(p_i * log2(p_i))
    """
    # Replace zeros with a very small number to avoid log2(0)
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
    Higher temperature leads to more random choices.
    """
    probabilities = np.asarray(probabilities).astype('float64')
    log_probs = np.log(probabilities) / temperature
    exp_probs = np.exp(log_probs - np.max(log_probs))
    softmax_probs = exp_probs / np.sum(exp_probs)
    
    # Ensure there are no NaNs, which can happen if all probabilities are zero.
    if np.isnan(softmax_probs).any():
        num_players = len(probabilities)
        return np.random.choice(num_players)

    return np.random.choice(len(softmax_probs), p=softmax_probs)
