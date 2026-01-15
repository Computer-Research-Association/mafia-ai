from typing import Dict, List, Tuple
from collections import defaultdict
from config import Role, EventType, Phase
from core.engine.state import GameEvent

# --- 1. Reward Configuration Constants ---
TIME_PENALTY = -0.1
DECEPTION_MULTIPLIER = 1.5

# Nested Dict Structure: [MyRole][EventType][TargetRole] -> RewardValue
# Default value is 0.0 to avoid if-else checks
REWARD_MATRIX = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

# === Matrix Initialization ===
# [MAFIA Rewards]
# - Execution
REWARD_MATRIX[Role.MAFIA][EventType.EXECUTE][Role.POLICE] = 10.0
REWARD_MATRIX[Role.MAFIA][EventType.EXECUTE][Role.DOCTOR] = 10.0
REWARD_MATRIX[Role.MAFIA][EventType.EXECUTE][Role.CITIZEN] = 3.0
REWARD_MATRIX[Role.MAFIA][EventType.EXECUTE][Role.MAFIA] = -20.0  # Killing teammate
# - Night Kill
# REWARD_MATRIX[Role.MAFIA][EventType.KILL][Role.POLICE] = 5.0
# REWARD_MATRIX[Role.MAFIA][EventType.KILL][Role.DOCTOR] = 5.0
# REWARD_MATRIX[Role.MAFIA][EventType.KILL][Role.CITIZEN] = 2.0
# (Usually killing is the means, not the end, but we can reward it)
REWARD_MATRIX[Role.MAFIA][EventType.KILL][Role.POLICE] = 5.0
REWARD_MATRIX[Role.MAFIA][EventType.KILL][Role.DOCTOR] = 5.0
REWARD_MATRIX[Role.MAFIA][EventType.KILL][Role.CITIZEN] = 2.0
REWARD_MATRIX[Role.MAFIA][EventType.KILL][Role.MAFIA] = -99.0  # Should be impossible

# [CITIZEN Team Rewards (Citizen, Police, Doctor)]
CIV_TEAM = [Role.CITIZEN, Role.POLICE, Role.DOCTOR]

for role in CIV_TEAM:
    # - Execution
    REWARD_MATRIX[role][EventType.EXECUTE][Role.MAFIA] = 10.0
    REWARD_MATRIX[role][EventType.EXECUTE][Role.CITIZEN] = -5.0
    REWARD_MATRIX[role][EventType.EXECUTE][Role.POLICE] = -10.0
    REWARD_MATRIX[role][EventType.EXECUTE][Role.DOCTOR] = -10.0
    
    # - Night Kill (Teammate death penalty)
    REWARD_MATRIX[role][EventType.KILL][Role.CITIZEN] = -1.0
    REWARD_MATRIX[role][EventType.KILL][Role.POLICE] = -2.0
    REWARD_MATRIX[role][EventType.KILL][Role.DOCTOR] = -2.0
    REWARD_MATRIX[role][EventType.KILL][Role.MAFIA] = 0.0 # Good but usually handled by execution

# [Role Specific Rewards]
# Police: Valid Investigation
# Note: POLICE_RESULT value usually indicates if target is mafia.
# We map the result logic inside the matrix if possible, or specialized signals.
# Here we assume TargetRole is the *actual* role of the investigated person.
REWARD_MATRIX[Role.POLICE][EventType.POLICE_RESULT][Role.MAFIA] = 3.0
REWARD_MATRIX[Role.POLICE][EventType.POLICE_RESULT][Role.CITIZEN] = 0.5

# Doctor: Successful Protection
# Note: This requires the event signal to carry information about successful defense.
# We will define a virtual event or use PROTECT if we know it was successful.
# For now, generic PROTECT reward (encouraging activity) or specific if implied.
REWARD_MATRIX[Role.DOCTOR][EventType.PROTECT][Role.POLICE] = 1.0
REWARD_MATRIX[Role.DOCTOR][EventType.PROTECT][Role.CITIZEN] = 0.5


class RewardManager:
    def __init__(self):
        """
        Initialize RewardManager.
        """
        # Track claims for the 'Deception Bonus'
        # Map: player_id -> last_claimed_role
        self.last_claims: Dict[int, Role] = {}

    def reset(self):
        """Reset internal state (claims)."""
        self.last_claims.clear()

    def calculate(self, game, start_idx: int = 0) -> Dict[int, float]:
        """
        Calculate rewards for all players based on events from start_idx to end.
        
        Args:
            game: Game engine instance (access to history, day, players)
            start_idx: Index in game.history to start processing from
        
        Returns:
            Dict[player_id, reward_value]
        """
        rewards = defaultdict(float)
        
        # 0. Apply Time Penalty (regardless of events)
        # Applied to all alive players
        current_time_penalty = TIME_PENALTY * game.day
        for p in game.players:
            if p.alive:
                rewards[p.id] += current_time_penalty
        
        # Win/Loss Bonus (Game End Check)
        # Check if the game officially ended in this step range
        # Use game.history to check for GAME_END event (not just signals)
        # But signals extraction handles events, so let's see.
        # It's better to check game status directly or check events.
        # Instructions say: "is_over 보상도 RewardManager가 관리하게 하면 더 완벽합니다."
        
        # Check for Game End events in the new batch
        new_events = game.history[start_idx:]
        
        # 1. Extract Signals from new events
        signals = self._extract_signals(new_events, game)

        # 2. Iterate and Summation
        for sig_type, target_role, actor_id, extra_info in signals:
            
            # --- Update Internal State (Claims) ---
            if sig_type == EventType.CLAIM and actor_id is not None:
                # extra_info should contain the claimed role
                if isinstance(extra_info, Role):
                    self.last_claims[actor_id] = extra_info
                continue 

            # --- Handle Game End Reward ---
            if sig_type == Phase.GAME_END:
                # extra_info is bool (Citizen Win?)
                citizen_win = extra_info
                for p in game.players:
                    is_citizen_team = p.role != Role.MAFIA
                    # If citizen_win is True: Citizens +10, Mafia -10
                    # If citizen_win is False: Mafia +10, Citizens -10
                    if citizen_win:
                        rewards[p.id] += 10.0 if is_citizen_team else -10.0
                    else:
                        rewards[p.id] += 10.0 if not is_citizen_team else -10.0
                continue

            # --- Calculate Rewards for each player ---
            for p in game.players:
                if not p.alive:
                    continue
                
                my_role = p.role
                
                # A. Base Lookup
                base_reward = REWARD_MATRIX[my_role][sig_type][target_role]
                
                # B. Apply Multipliers (Mafia Deception Bonus)
                # Risk/Reward multiplier: Increases both penalty and reward.
                multiplier = 1.0
                if my_role == Role.MAFIA:
                    # Check if this mafia is currently claiming POLICE
                    current_claim = self.last_claims.get(p.id)
                    if current_claim == Role.POLICE:
                        multiplier = DECEPTION_MULTIPLIER
                
                # C. Final Summation
                rewards[p.id] += base_reward * multiplier

        return dict(rewards)

    def _extract_signals(self, events: List[GameEvent], game) -> List[Tuple[EventType, Role, int, any]]:
        """
        Convert raw GameEvents into structured signals: (EventType, TargetRole, ActorID, Extra/Value)
        """
        signals = []
        
        for event in events:
            # 1. Resolve Target Role
            target_role = None
            if event.target_id != -1 and 0 <= event.target_id < len(game.players):
                target_role = game.players[event.target_id].role

            # 2. Game End Special Handling
            if event.phase == Phase.GAME_END:
                # Treat Phase.GAME_END as a signal type for internal logic, 
                # though it's technically a Phase, not EventType in the enum 
                # (but GameEvent struct uses phase field). 
                # Ideally check event_type, but engine.game.py might emit generic system message for end.
                # Let's check how game_end is logged. 
                # game.py: event = GameEvent(..., phase=Phase.GAME_END, ...)
                # It usually doesn't have a specific event_type for "WIN/LOSS", it uses system message or similar.
                # Assuming the event carries the winner info in `value`.
                signals.append((Phase.GAME_END, None, -1, event.value))
                continue

            # 3. Standard Signals for Matrix Lookup
            if event.event_type in [EventType.EXECUTE, EventType.KILL, 
                                   EventType.POLICE_RESULT, EventType.PROTECT]:
                signals.append((event.event_type, target_role, event.actor_id, event.value))
            
            # 4. Internal State Update Signals (Claim)
            elif event.event_type == EventType.CLAIM:
                signals.append((EventType.CLAIM, None, event.actor_id, event.value))
                
        return signals

