"""
Core Bayesian Inference Engine for the Rule-Based Agent.
"""
import numpy as np
from typing import List, Optional

from core.engine.state import GameStatus
from .constants import RBAEventType
from .strategies import get_contextual_likelihood, BASE_LIKELIHOODS


class BayesianInferenceEngine:
    """
    Manages the probabilistic model for identifying Mafia members.
    P(M|E) propto P(E|M) * P(M)
    """
    def __init__(self, num_players: int, agent_id: int, agent_team: str, mafia_count: int, mafia_team_ids: List[int] = []):
        self.num_players = num_players
        self.agent_id = agent_id
        self.agent_team = agent_team
        self.initial_mafia_count = mafia_count
        self.remaining_mafia_count = float(mafia_count)
        
        self.prior = self._initialize_prior(mafia_count, mafia_team_ids)
        self.posterior = np.copy(self.prior)
        
        # Correlation matrix to track player alignments
        self.correlation = np.identity(self.num_players) * 0.1

    def _initialize_prior(self, mafia_count: int, mafia_team_ids: List[int]) -> np.ndarray:
        """
        Initializes the prior probability distribution P(M) for each player being Mafia.
        """
        if self.agent_team == 'mafia':
            prior = np.zeros(self.num_players)
            for i in mafia_team_ids:
                prior[i] = 1.0
        else: # Town roles
            initial_prob = mafia_count / self.num_players if self.num_players > 0 else 0
            prior = np.full(self.num_players, initial_prob)

        prior[self.agent_id] = 1.0 if self.agent_team == 'mafia' else 0.0
        return self._normalize(prior)

    def update(self, event_type: RBAEventType, actor_id: int, target_id: Optional[int] = None, status: Optional[GameStatus] = None):
        """
        Updates the posterior probability based on a new event (evidence E).
        """
        if actor_id >= self.num_players:
            return

        # Get the contextual likelihood P(E|M)
        if status:
            survivor_count = sum(1 for p in status.players if p.alive)
            # Pass the engine itself for correlation analysis
            likelihood = get_contextual_likelihood(event_type, day=status.day, survivor_count=survivor_count, actor_id=actor_id, engine=self)
        else:
            likelihood = BASE_LIKELIHOODS.get(event_type, 1.0)

        # Update the belief for the actor player
        self.posterior[actor_id] *= likelihood

        # Normalize the posterior to make it a valid probability distribution
        self.posterior = self._normalize(self.posterior)
        self.prior = np.copy(self.posterior)
        
    def update_correlation(self, p1: int, p2: int, strength: float):
        """
        Increases the correlation between two players.
        """
        if p1 >= self.num_players or p2 >= self.num_players: return
        
        # Increase correlation symmetrically and clamp
        new_corr12 = self.correlation[p1, p2] + strength
        new_corr21 = self.correlation[p2, p1] + strength
        self.correlation[p1, p2] = min(new_corr12, 1.0)
        self.correlation[p2, p1] = min(new_corr21, 1.0)

    def set_player_probability(self, player_id: int, is_mafia: bool):
        """
        Hard sets a player's mafia probability based on definitive information.
        """
        if player_id >= self.num_players:
            return

        was_known_mafia = self.posterior[player_id] == 1.0
        if is_mafia and not was_known_mafia:
            if self.posterior[player_id] < 0.99:
                 self.remaining_mafia_count -= 1

        self.posterior[player_id] = 1.0 if is_mafia else 0.0
        
        self.posterior = self._normalize(self.posterior)
        self.prior = np.copy(self.posterior)

    def get_mafia_probabilities(self) -> np.ndarray:
        """
        Returns the current posterior probability distribution.
        """
        return self.posterior

    def _normalize(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Normalizes the array so that the sum of probabilities equals the number of
        remaining mafias. This version includes safeguards against numerical instability.
        """
        # Safeguard against NaN/inf values from previous calculations
        if not np.all(np.isfinite(probabilities)):
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)

        fixed_indices = np.where((probabilities == 0.0) | (probabilities == 1.0))[0]
        fixed_prob_sum = np.sum(probabilities[fixed_indices])
        
        target_sum_for_uncertain = self.remaining_mafia_count - fixed_prob_sum
        if target_sum_for_uncertain < 0:
            target_sum_for_uncertain = 0

        uncertain_indices = np.where((probabilities > 0.0) & (probabilities < 1.0))[0]
        
        if len(uncertain_indices) == 0:
            return np.clip(probabilities, 0, 1)

        current_uncertain_sum = np.sum(probabilities[uncertain_indices])
        
        # Avoid division by zero or very small numbers
        if current_uncertain_sum > 1e-9:
            scale_factor = target_sum_for_uncertain / current_uncertain_sum
            probabilities[uncertain_indices] *= scale_factor
        elif target_sum_for_uncertain > 0:
            # If current sum is zero but it should be positive, distribute evenly
            probabilities[uncertain_indices] = target_sum_for_uncertain / len(uncertain_indices)

        # Final clipping to ensure all values are valid probabilities
        return np.clip(probabilities, 0, 1)

    def reset_and_replay(self, mafia_count: int, mafia_team_ids: List[int]):
        """Resets the engine for a new round of history replay."""
        self.initial_mafia_count = mafia_count
        self.remaining_mafia_count = float(mafia_count)
        self.prior = self._initialize_prior(mafia_count, mafia_team_ids)
        self.posterior = np.copy(self.prior)
        self.correlation = np.identity(self.num_players) * 0.1