"""
Core Bayesian Inference Engine for the Rule-Based Agent.
"""
import numpy as np
from typing import List, Optional
from .constants import LIKELIHOOD_MATRIX, RBAEventType

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

    def _initialize_prior(self, mafia_count: int, mafia_team_ids: List[int]) -> np.ndarray:
        """
        Initializes the prior probability distribution P(M) for each player being Mafia.
        - If this agent is Mafia, it knows its teammates.
        - If this agent is Town, it assumes an initial probability based on the number of Mafias.
        """
        if self.agent_team == 'mafia':
            # Mafia knows who other mafias are.
            prior = np.zeros(self.num_players)
            for i in mafia_team_ids:
                prior[i] = 1.0
        else: # Town roles
            # Townies start with a uniform prior based on mafia count.
            initial_prob = mafia_count / self.num_players if self.num_players > 0 else 0
            prior = np.full(self.num_players, initial_prob)

        # An agent knows its own role is not mafia (unless it is)
        prior[self.agent_id] = 1.0 if self.agent_team == 'mafia' else 0.0
        
        # This initial normalization sets the stage.
        return self._normalize(prior)

    def update(self, event_type: RBAEventType, actor_id: int, target_id: Optional[int] = None):
        """
        Updates the posterior probability based on a new event (evidence E).
        P(M_i|E) propto P(E|M_i) * P(M_i) for a target player i.
        """
        if actor_id >= self.num_players:
            return

        # Get the likelihood P(E|M) from the matrix
        likelihood = LIKELIHOOD_MATRIX.get(event_type, 1.0)

        # Update the belief for the actor player
        self.posterior[actor_id] *= likelihood

        # If the event has a target, we might update them as well
        if target_id is not None and target_id < self.num_players:
            if event_type == RBAEventType.PLAYER_WAS_VOTED_FOR:
                 self.posterior[target_id] *= LIKELIHOOD_MATRIX.get(RBAEventType.PLAYER_WAS_VOTED_FOR, 1.0)

        # Normalize the posterior to make it a valid probability distribution
        self.posterior = self._normalize(self.posterior)
        self.prior = np.copy(self.posterior)

    def set_player_probability(self, player_id: int, is_mafia: bool):
        """
        Hard sets a player's mafia probability based on definitive information
        (e.g., police scan, post-death role reveal).
        """
        if player_id >= self.num_players:
            return

        # If a new mafia is confirmed, decrement the remaining count.
        was_known_mafia = self.posterior[player_id] == 1.0
        if is_mafia and not was_known_mafia:
            # This check is crucial to only decrement once per mafia.
            if self.posterior[player_id] < 0.99: # Allow for floating point inaccuracies
                 self.remaining_mafia_count -= 1

        # Set probability and ensure it's not altered during normalization
        self.posterior[player_id] = 1.0 if is_mafia else 0.0
        
        # Re-normalize probabilities for all other players
        self.posterior = self._normalize(self.posterior)
        self.prior = np.copy(self.posterior)

    def get_mafia_probabilities(self) -> np.ndarray:
        """
        Returns the current posterior probability distribution P(M|E_1, ..., E_n).
        """
        return self.posterior

    def _normalize(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Normalizes the array so that the sum of probabilities equals the number of
        remaining mafias.
        """
        # Players with known identities (0 or 1) are considered "fixed"
        fixed_indices = np.where((probabilities == 0.0) | (probabilities == 1.0))[0]
        fixed_prob_sum = np.sum(probabilities[fixed_indices])
        
        # The sum of probabilities for uncertain players should be the remainder.
        target_sum_for_uncertain = self.remaining_mafia_count - fixed_prob_sum
        
        # Clamp target sum to be non-negative.
        if target_sum_for_uncertain < 0:
            target_sum_for_uncertain = 0

        uncertain_indices = np.where((probabilities > 0.0) & (probabilities < 1.0))[0]
        
        # Avoid division by zero if there are no uncertain players.
        if len(uncertain_indices) == 0:
            return probabilities

        current_uncertain_sum = np.sum(probabilities[uncertain_indices])
        
        if current_uncertain_sum > 0:
            scale_factor = target_sum_for_uncertain / current_uncertain_sum
            probabilities[uncertain_indices] *= scale_factor
        elif target_sum_for_uncertain > 0 and len(uncertain_indices) > 0:
            # If current sum is 0, but should be > 0, distribute equally.
            probabilities[uncertain_indices] = target_sum_for_uncertain / len(uncertain_indices)

        # Final check to clamp probabilities between 0 and 1.
        probabilities = np.clip(probabilities, 0, 1)
        return probabilities

    def reset_and_replay(self, mafia_count: int, mafia_team_ids: List[int]):
        """Resets the engine for a new round of history replay."""
        self.initial_mafia_count = mafia_count
        self.remaining_mafia_count = float(mafia_count)
        self.prior = self._initialize_prior(mafia_count, mafia_team_ids)
        self.posterior = np.copy(self.prior)