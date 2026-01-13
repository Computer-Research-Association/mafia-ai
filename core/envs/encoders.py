from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from config import config, Role, Phase, EventType
from core.engine.state import GameEvent

class BaseEncoder(ABC):
    """
    Abstract base class for observation encoders.
    """
    @abstractmethod
    def encode(self, game, player_id: int) -> np.ndarray:
        """
        Encodes the game state into an observation vector for the specified player.
        
        Args:
            game: The MafiaGame instance.
            player_id: The ID of the player to generate the observation for.
            
        Returns:
            np.ndarray: The encoded observation vector.
        """
        pass
    
    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """
        Returns the dimension of the observation vector.
        """
        pass


class MDPEncoder(BaseEncoder):
    """
    Encoder for MLP backbone.
    Generates a dense vector (286 dim) including cumulative maps (Vote, Attack, Vouch).
    """
    def __init__(self):
        self._dim = 286

    @property
    def observation_dim(self) -> int:
        return self._dim

    def encode(self, game, player_id: int) -> np.ndarray:
        status = game.get_game_status(player_id)
        
        # === [Maps Generation] ===
        # 1. Vote Map (8x8)
        vote_map = np.zeros((8, 8), dtype=np.float32)
        # 2. Attack Map (8x8)
        attack_map = np.zeros((8, 8), dtype=np.float32)
        # 3. Vouch Map (8x8)
        vouch_map = np.zeros((8, 8), dtype=np.float32)
        # 4. Role Claim Map (8x5)
        claim_map = np.zeros((8, 5), dtype=np.float32)
        
        # 5. Alive Map (8)
        alive_vec = np.array([1.0 if p.alive else 0.0 for p in game.players], dtype=np.float32)

        # History Traversal (Cumulative)
        for event in status.action_history:
            actor = event.actor_id
            target = event.target_id
            
            # Skip invalid actor/target indices (0~7 only)
            if actor < 0 or actor >= 8:
                continue

            # Role Claim Update
            if event.event_type == EventType.CLAIM and isinstance(event.value, Role):
                claim_map[actor][int(event.value)] = 1.0
            
            if target is None or target < 0 or target >= 8:
                continue

            # Cumulative Updates
            if event.event_type == EventType.VOTE:
                vote_map[actor][target] += 1.0
            elif event.event_type in [EventType.KILL, EventType.POLICE_RESULT]:
                attack_map[actor][target] += 1.0
            elif event.event_type in [EventType.PROTECT]:
                vouch_map[actor][target] += 1.0

        # Normalization (Max Scaling)
        if np.max(vote_map) > 0: vote_map /= np.max(vote_map)
        if np.max(attack_map) > 0: attack_map /= np.max(attack_map)
        if np.max(vouch_map) > 0: vouch_map /= np.max(vouch_map)

        # 1. Self Info (12)
        # ID One-hot (8)
        id_vec = np.zeros(8, dtype=np.float32)
        id_vec[player_id] = 1.0

        # Role One-hot (4)
        role_vec = np.zeros(4, dtype=np.float32)
        role_vec[int(status.my_role)] = 1.0

        # 2. Game Context (4)
        # Day (1)
        day_vec = np.array([status.day / float(config.game.MAX_DAYS)], dtype=np.float32)

        # Phase One-hot (3)
        phase_vec = np.zeros(3, dtype=np.float32)
        if status.phase == Phase.DAY_DISCUSSION:
            phase_vec[0] = 1.0
        elif status.phase == Phase.DAY_VOTE:
            phase_vec[1] = 1.0
        else:  # Execute or Night
            phase_vec[2] = 1.0

        # 3. Last Event (30)
        last_event = None
        if status.action_history:
            last_event = status.action_history[-1]

        if last_event:
            # Actor ID (9)
            actor_vec = np.zeros(9, dtype=np.float32)
            if last_event.actor_id == -1:
                actor_vec[8] = 1.0
            else:
                actor_vec[last_event.actor_id] = 1.0

            # Target ID (9)
            target_vec = np.zeros(9, dtype=np.float32)
            if last_event.target_id is None or last_event.target_id == -1:
                target_vec[8] = 1.0
            else:
                target_vec[last_event.target_id] = 1.0

            # Value/Role (5)
            value_vec = np.zeros(5, dtype=np.float32)
            if last_event.value is None:
                value_vec[4] = 1.0
            elif isinstance(last_event.value, Role):
                value_vec[int(last_event.value)] = 1.0
            else:
                value_vec[4] = 1.0

            # Event Type (7)
            type_vec = np.zeros(7, dtype=np.float32)
            try:
                t_idx = int(last_event.event_type) - 1
                if 0 <= t_idx < 7:
                    type_vec[t_idx] = 1.0
            except:
                pass

        else:
            actor_vec = np.zeros(9, dtype=np.float32)
            actor_vec[8] = 1.0
            target_vec = np.zeros(9, dtype=np.float32)
            target_vec[8] = 1.0
            value_vec = np.zeros(5, dtype=np.float32)
            value_vec[4] = 1.0
            type_vec = np.zeros(7, dtype=np.float32)

        # Concatenate all
        obs = np.concatenate(
            [
                id_vec,       # 8
                role_vec,     # 4
                day_vec,      # 1
                phase_vec,    # 3
                actor_vec,    # 9
                target_vec,   # 9
                value_vec,    # 5
                type_vec,     # 7
                # === Maps ===
                vote_map.flatten(),    # 64
                attack_map.flatten(),  # 64
                vouch_map.flatten(),   # 64
                claim_map.flatten(),   # 40
                alive_vec              # 8
            ]
        )  
        return obs


class POMDPEncoder(BaseEncoder):
    """
    Encoder for RNN backbone.
    Generates a slim vector (54 dim) containing only current event and status.
    Maps are excluded to rely on RNN memory.
    """
    def __init__(self):
        self._dim = 54  # 46 (Base) + 8 (Alive)

    @property
    def observation_dim(self) -> int:
        return self._dim

    def encode(self, game, player_id: int) -> np.ndarray:
        status = game.get_game_status(player_id)
        
        # 1. Self Info (12)
        id_vec = np.zeros(8, dtype=np.float32)
        id_vec[player_id] = 1.0

        role_vec = np.zeros(4, dtype=np.float32)
        role_vec[int(status.my_role)] = 1.0

        # 2. Game Context (4)
        day_vec = np.array([status.day / float(config.game.MAX_DAYS)], dtype=np.float32)

        phase_vec = np.zeros(3, dtype=np.float32)
        if status.phase == Phase.DAY_DISCUSSION:
            phase_vec[0] = 1.0
        elif status.phase == Phase.DAY_VOTE:
            phase_vec[1] = 1.0
        else:
            phase_vec[2] = 1.0

        # 3. Last Event (30)
        last_event = None
        if status.action_history:
            last_event = status.action_history[-1]

        if last_event:
            actor_vec = np.zeros(9, dtype=np.float32)
            if last_event.actor_id == -1:
                actor_vec[8] = 1.0
            else:
                actor_vec[last_event.actor_id] = 1.0

            target_vec = np.zeros(9, dtype=np.float32)
            if last_event.target_id is None or last_event.target_id == -1:
                target_vec[8] = 1.0
            else:
                target_vec[last_event.target_id] = 1.0

            value_vec = np.zeros(5, dtype=np.float32)
            if last_event.value is None:
                value_vec[4] = 1.0
            elif isinstance(last_event.value, Role):
                value_vec[int(last_event.value)] = 1.0
            else:
                value_vec[4] = 1.0

            type_vec = np.zeros(7, dtype=np.float32)
            try:
                t_idx = int(last_event.event_type) - 1
                if 0 <= t_idx < 7:
                    type_vec[t_idx] = 1.0
            except:
                pass

        else:
            actor_vec = np.zeros(9, dtype=np.float32)
            actor_vec[8] = 1.0
            target_vec = np.zeros(9, dtype=np.float32)
            target_vec[8] = 1.0
            value_vec = np.zeros(5, dtype=np.float32)
            value_vec[4] = 1.0
            type_vec = np.zeros(7, dtype=np.float32)

        # 4. Alive Map (8) - Essential for decision making
        alive_vec = np.array([1.0 if p.alive else 0.0 for p in game.players], dtype=np.float32)

        # Concatenate
        obs = np.concatenate(
            [
                id_vec,       # 8
                role_vec,     # 4
                day_vec,      # 1
                phase_vec,    # 3
                actor_vec,    # 9
                target_vec,   # 9
                value_vec,    # 5
                type_vec,     # 7
                alive_vec     # 8
            ]
        )
        return obs
