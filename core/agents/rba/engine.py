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
    def __init__(self, num_players: int, agent_id: int, agent_team: str, player_roles: List[str], known_mafia_ids: List[int]):
        self.num_players = num_players
        self.agent_id = agent_id
        self.agent_team = agent_team
        
        self.prior = self._initialize_prior(player_roles, known_mafia_ids)
        self.posterior = np.copy(self.prior)

    def _initialize_prior(self, player_roles: List[str], known_mafia_ids: List[int]) -> np.ndarray:
        """
        Initializes the prior probability distribution P(M) for each player being Mafia.
        - If this agent is Mafia, it knows its teammates.
        - If this agent is Town, it assumes an initial probability based on the number of Mafias.
        """
        mafia_count = player_roles.count('mafia')
        
        if self.agent_team == 'mafia':
            prior = np.zeros(self.num_players)
            for i in range(self.num_players):
                if player_roles[i] == 'mafia':
                    prior[i] = 1.0
        else: # Town roles
            initial_prob = mafia_count / self.num_players if self.num_players > 0 else 0
            prior = np.full(self.num_players, initial_prob)
            # Town agents also know who the revealed mafias are
            for mafia_id in known_mafia_ids:
                prior[mafia_id] = 1.0

        # An agent knows its own role is not mafia (unless it is)
        prior[self.agent_id] = 1.0 if self.agent_team == 'mafia' else 0.0
        
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
        Normalizes the array so that the sum of probabilities of uncertain players equals
        the expected number of remaining mafias.
        """
        fixed_prob_sum = 0
        fixed_indices = []

        # An agent knows its own identity
        fixed_indices.append(self.agent_id)
        fixed_prob_sum += probabilities[self.agent_id]
        
        # Identify players with known identities (0 or 1)
        for i in range(self.num_players):
            if i != self.agent_id and (probabilities[i] == 0.0 or probabilities[i] == 1.0):
                fixed_indices.append(i)
                fixed_prob_sum += probabilities[i]

        # Calculate sum of probabilities for players with uncertain identities
        uncertain_indices = [i for i in range(self.num_players) if i not in fixed_indices]
        uncertain_sum = np.sum(probabilities[uncertain_indices])

        if uncertain_sum > 0:
            # We can't know the true mafia count, so we just normalize the rest to sum to 1
            # and then scale by a factor. A simple normalization to sum to 1 is a good start.
            scale = 1.0 - fixed_prob_sum
            if scale < 0: scale = 0 # Should not happen in a logical game
            
            probabilities[uncertain_indices] = (probabilities[uncertain_indices] / uncertain_sum) * scale

        # Final check to avoid floating point errors leading to negative probabilities
        probabilities[probabilities < 0] = 0
        return probabilities

    def reset_and_replay(self, history: List, player_roles: List[str], known_mafia_ids: List[int]):
        """Resets the engine and replays the history to rebuild the posterior."""
        self.prior = self._initialize_prior(player_roles, known_mafia_ids)
        self.posterior = np.copy(self.prior)
        # This is where you would loop through history and call self.update() and self.set_player_probability()
        # For now, this is handled in the agent itself.
        pass