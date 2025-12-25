"""
Rational Rule-Based Agent (RBA) for Mafia Game

This module implements a unified rational agent that makes decisions based purely
on logical deduction and belief scores, without personality-based randomness.
"""

from typing import List, Set
import numpy as np
import config
from core.characters.base import BaseCharacter, softmax


class RationalCharacter(BaseCharacter):
    """
    Rational Agent that makes decisions based on logical deduction rules.
    
    The agent maintains a belief matrix about each player's role probability
    and updates it based on observed events using rational deduction rules.
    """
    
    def __init__(self, player_id: int, role: int = config.ROLE_CITIZEN):
        super().__init__(player_id, role)
        
        # Additional tracking for rational decisions
        self.accusation_history = []  # List of (accuser_id, accused_id, day) tuples
        self.execution_history = []   # List of (player_id, team, day) tuples
        self.healed_players = set()   # Players I healed (for doctors)
        self.no_death_nights = []     # Nights when no one died
        
        # Initialize belief matrix with uniform distribution
        initial_belief = 25.0  # 25% for each role initially (uniform)
        self.belief = np.full((config.PLAYER_COUNT, 4), initial_belief, dtype=np.float32)
        
        # Set self-knowledge with certainty
        self.belief[self.id] = 0.0
        self.belief[self.id, self.role] = 100.0
    
    def update_belief(self, game_status: dict):
        """
        Update belief matrix based on observed game events using rational deduction rules.
        
        Rules implemented:
        1. Self-Knowledge (Citizen View): If accused while innocent, accuser likely Mafia
        2. Threat Detection (Mafia View): If accused while Mafia, accuser likely Police
        3. Deduction from Execution: If accuser's target was executed and revealed as Citizen, accuser likely Mafia
        4. Doctor's Logic: If healed someone and no death occurred, target likely Citizen
        
        Args:
            game_status: Dictionary containing game state information including:
                - 'claims': List of (speaker_id, target_id) tuples
                - 'alive_players': List of alive player IDs
                - 'execution_result': Optional (executed_id, team_alignment, day)
                - 'night_result': Optional info about night deaths
        """
        if not self.alive:
            return
        
        # Extract information from game status
        claims = game_status.get('claims', [])
        alive_players = game_status.get('alive_players', [])
        execution_result = game_status.get('execution_result', None)
        night_result = game_status.get('night_result', None)
        
        # Process new accusations/claims
        for accuser_id, accused_id in claims:
            if accuser_id == self.id or accused_id == -1:
                continue
            
            # Store accusation history
            current_day = game_status.get('day', 1)
            self.accusation_history.append((accuser_id, accused_id, current_day))
            
            # Rule 1: Self-Knowledge (Citizen View)
            # IF I am accused AND I am NOT Mafia
            # THEN accuser is likely Mafia (false accusation)
            if accused_id == self.id and self.role != config.ROLE_MAFIA:
                self._apply_belief_update(
                    accuser_id, 
                    role_idx=config.ROLE_MAFIA, 
                    delta=15.0,
                    reason="false_accusation_against_me"
                )
            
            # Rule 2: Threat Detection (Mafia View)
            # IF I am Mafia AND someone accuses me
            # THEN accuser is likely Police (high threat)
            if accused_id == self.id and self.role == config.ROLE_MAFIA:
                self._apply_belief_update(
                    accuser_id,
                    role_idx=config.ROLE_POLICE,
                    delta=20.0,
                    reason="accused_me_as_mafia"
                )
        
        # Rule 3: Deduction from Execution
        # IF Player A accused Player B AND Player B was executed and revealed as Citizen Team
        # THEN Player A is likely Mafia
        if execution_result:
            executed_id, team_alignment, day = execution_result
            
            # Store execution history
            self.execution_history.append((executed_id, team_alignment, day))
            
            # Clear beliefs about executed player
            self.belief[executed_id] = 0.0
            
            # Update team alignment beliefs
            if team_alignment == "CITIZEN":
                # Executed player was on Citizen Team
                self.belief[executed_id, config.ROLE_CITIZEN] = 100.0
                self.belief[executed_id, config.ROLE_POLICE] = 0.0
                self.belief[executed_id, config.ROLE_DOCTOR] = 0.0
                self.belief[executed_id, config.ROLE_MAFIA] = 0.0
                
                # Find who accused this player
                for accuser_id, accused_id, acc_day in self.accusation_history:
                    if accused_id == executed_id and acc_day == day:
                        # Accuser pushed for a Citizen's execution
                        self._apply_belief_update(
                            accuser_id,
                            role_idx=config.ROLE_MAFIA,
                            delta=25.0,
                            reason=f"accused_citizen_{executed_id}"
                        )
            
            elif team_alignment == "MAFIA":
                # Executed player was on Mafia Team
                self.belief[executed_id, config.ROLE_MAFIA] = 100.0
                self.belief[executed_id, config.ROLE_CITIZEN] = 0.0
                self.belief[executed_id, config.ROLE_POLICE] = 0.0
                self.belief[executed_id, config.ROLE_DOCTOR] = 0.0
                
                # Find who accused this player
                for accuser_id, accused_id, acc_day in self.accusation_history:
                    if accused_id == executed_id and acc_day == day:
                        # Accuser found a Mafia - likely Police or good citizen
                        if self.role != config.ROLE_MAFIA:
                            self._apply_belief_update(
                                accuser_id,
                                role_idx=config.ROLE_POLICE,
                                delta=20.0,
                                reason=f"found_mafia_{executed_id}"
                            )
                            # Decrease Mafia probability
                            self._apply_belief_update(
                                accuser_id,
                                role_idx=config.ROLE_MAFIA,
                                delta=-30.0,
                                reason=f"found_mafia_{executed_id}"
                            )
        
        # Rule 4: Doctor's Logic
        # IF I am Doctor AND I healed someone AND no death occurred
        # THEN target is likely Citizen (Mafia target)
        if self.role == config.ROLE_DOCTOR and night_result:
            no_death = night_result.get('no_death', False)
            last_healed = night_result.get('last_healed', -1)
            
            if no_death and last_healed != -1:
                self.healed_players.add(last_healed)
                self.no_death_nights.append(game_status.get('day', 1))
                
                # Increase probability that healed player is Citizen team
                self._apply_belief_update(
                    last_healed,
                    role_idx=config.ROLE_CITIZEN,
                    delta=15.0,
                    reason="successful_heal"
                )
                # Decrease Mafia probability
                self._apply_belief_update(
                    last_healed,
                    role_idx=config.ROLE_MAFIA,
                    delta=-20.0,
                    reason="successful_heal"
                )
        
        # Normalize beliefs to keep them in reasonable ranges
        self._normalize_beliefs(alive_players)
    
    def _apply_belief_update(
        self, 
        player_id: int, 
        role_idx: int, 
        delta: float,
        reason: str = ""
    ):
        """
        Apply a belief update for a specific player and role.
        
        Args:
            player_id: The player whose belief to update
            role_idx: The role index (0: Citizen, 1: Police, 2: Doctor, 3: Mafia)
            delta: The amount to change the belief (positive or negative)
            reason: Optional reason for logging
        """
        if player_id == self.id:
            return  # Never update beliefs about self
        
        # Apply update with bounds
        old_value = self.belief[player_id, role_idx]
        self.belief[player_id, role_idx] = np.clip(
            old_value + delta,
            -100.0,
            100.0
        )
    
    def _normalize_beliefs(self, alive_players: List[int]):
        """
        Normalize beliefs to maintain consistency and prevent extreme values.
        
        For each player, ensure:
        - Dead players have no belief
        - Self belief remains certain
        - Sum of probabilities doesn't explode
        """
        for player_id in range(config.PLAYER_COUNT):
            if player_id == self.id:
                continue  # Keep self-knowledge certain
            
            if player_id not in alive_players:
                # Dead players retain their last known state
                continue
            
            # Soft normalization: prevent extreme values
            total = np.sum(self.belief[player_id])
            if total > 200:  # Too high
                self.belief[player_id] *= (200 / total)
            elif total < -200:  # Too negative
                self.belief[player_id] *= (-200 / total)
    
    def decide_claim(self, players: List["BaseCharacter"], current_day: int = 1) -> int:
        """
        Decide which player to accuse during discussion phase.
        
        Strategy based on role:
        - Police: Accuse confirmed Mafia (from investigation)
        - Citizen/Doctor: Accuse highest suspected Mafia with high confidence
        - Mafia: Accuse suspected Police or random citizen strategically
        
        Args:
            players: List of all players
            current_day: Current game day
            
        Returns:
            Player ID to accuse, or -1 to remain silent
        """
        alive_ids = self._get_alive_ids(players, exclude_me=True)
        if not alive_ids:
            return -1
        
        # === Police Strategy ===
        if self.role == config.ROLE_POLICE:
            # If confirmed Mafia exists, accuse them
            for mafia_id in self.confirmed_mafia:
                if mafia_id in alive_ids:
                    self.should_reveal = True
                    return mafia_id
            
            # Day 1-2: Stay quiet to gather information
            if current_day <= 2:
                return -1
            
            # Later: Accuse highest Mafia suspect with confidence > 70
            mafia_scores = [(pid, self.belief[pid, config.ROLE_MAFIA]) for pid in alive_ids]
            mafia_scores.sort(key=lambda x: x[1], reverse=True)
            
            if mafia_scores[0][1] > 70:
                return mafia_scores[0][0]
            
            return -1
        
        # === Citizen/Doctor Strategy ===
        elif self.role in [config.ROLE_CITIZEN, config.ROLE_DOCTOR]:
            # Day 1-2: Stay quiet, observe
            if current_day <= 2:
                return -1
            
            # Accuse only with very high confidence (> 75)
            mafia_scores = [(pid, self.belief[pid, config.ROLE_MAFIA]) for pid in alive_ids]
            mafia_scores.sort(key=lambda x: x[1], reverse=True)
            
            if mafia_scores[0][1] > 75:
                return mafia_scores[0][0]
            
            return -1
        
        # === Mafia Strategy ===
        elif self.role == config.ROLE_MAFIA:
            # Exclude other Mafia from targets
            non_mafia_alive = [
                pid for pid in alive_ids
                if players[pid].role != config.ROLE_MAFIA
            ]
            
            if not non_mafia_alive:
                return -1
            
            # Day 1: Stay quiet
            if current_day == 1:
                return -1
            
            # Priority 1: Accuse suspected Police (preemptive strike)
            police_suspects = [
                (pid, self.belief[pid, config.ROLE_POLICE]) 
                for pid in non_mafia_alive
            ]
            police_suspects.sort(key=lambda x: x[1], reverse=True)
            
            if police_suspects[0][1] > 60:
                return police_suspects[0][0]
            
            # Priority 2: Random accusation to create confusion (40% chance)
            if np.random.random() < 0.4:
                return np.random.choice(non_mafia_alive)
            
            return -1
        
        return -1
    
    def decide_vote(self, players: List["BaseCharacter"], current_day: int = 1) -> int:
        """
        Decide who to vote for execution.
        
        Strategy:
        - Vote for player with highest Mafia suspicion
        - Consider confidence threshold based on game day
        - Mafia votes strategically to eliminate threats
        
        Args:
            players: List of all players
            current_day: Current game day
            
        Returns:
            Player ID to vote for, or -1 to abstain
        """
        alive_ids = self._get_alive_ids(players, exclude_me=True)
        if not alive_ids:
            return -1
        
        # Get Mafia suspicion scores for alive players
        mafia_scores = np.array([self.belief[pid, config.ROLE_MAFIA] for pid in alive_ids])
        max_score = np.max(mafia_scores)
        
        # === Citizen Team Strategy ===
        if self.role != config.ROLE_MAFIA:
            # Day 1-2: High threshold (> 60)
            if current_day <= 2:
                if max_score > 60:
                    candidates = [pid for pid in alive_ids if self.belief[pid, config.ROLE_MAFIA] > 60]
                    scores = np.array([self.belief[pid, config.ROLE_MAFIA] for pid in candidates])
                    return self._select_target_softmax(candidates, scores, temperature=0.1)
                return -1
            
            # Day 3+: Lower threshold (> 40)
            if max_score > 40:
                return self._select_target_softmax(alive_ids, mafia_scores, temperature=0.1)
            
            return -1
        
        # === Mafia Strategy ===
        else:
            # Exclude other Mafia
            non_mafia_alive = [
                pid for pid in alive_ids
                if players[pid].role != config.ROLE_MAFIA
            ]
            
            if not non_mafia_alive:
                return -1
            
            # Vote to eliminate threats: Police > Doctor > Others
            threat_scores = np.array([
                self.belief[pid, config.ROLE_POLICE] * 2.0 +  # Police is highest threat
                self.belief[pid, config.ROLE_DOCTOR] * 1.5    # Doctor is moderate threat
                for pid in non_mafia_alive
            ])
            
            return self._select_target_softmax(non_mafia_alive, threat_scores, temperature=0.1)
    
    def decide_night_action(
        self, 
        players: List["BaseCharacter"], 
        current_role: int
    ) -> int:
        """
        Decide night action based on role.
        
        Args:
            players: List of all players
            current_role: Current role (should match self.role)
            
        Returns:
            Target player ID for night action
        """
        alive_ids = self._get_alive_ids(players, exclude_me=False)
        if not alive_ids:
            return -1
        
        # === Mafia: Kill highest threat ===
        if current_role == config.ROLE_MAFIA:
            candidates = [
                pid for pid in alive_ids
                if pid != self.id and players[pid].role != config.ROLE_MAFIA
            ]
            
            if not candidates:
                return -1
            
            # Priority: Revealed Police > Suspected Police > Doctor > Others
            threat_scores = np.array([
                (100.0 if players[pid].should_reveal and players[pid].role == config.ROLE_POLICE else 0.0) +
                self.belief[pid, config.ROLE_POLICE] * 2.0 +
                self.belief[pid, config.ROLE_DOCTOR] * 1.2
                for pid in candidates
            ])
            
            return self._select_target_softmax(candidates, threat_scores, temperature=0.1)
        
        # === Doctor: Protect highest value target ===
        elif current_role == config.ROLE_DOCTOR:
            candidates = alive_ids
            
            if not candidates:
                return -1
            
            # Priority: Revealed Police > Suspected Police > Self
            protection_scores = np.array([
                (100.0 if players[pid].should_reveal and players[pid].role == config.ROLE_POLICE else 0.0) +
                self.belief[pid, config.ROLE_POLICE] * 1.5 +
                (10.0 if pid == self.id else 0.0)  # Can protect self
                for pid in candidates
            ])
            
            return self._select_target_softmax(candidates, protection_scores, temperature=0.1)
        
        # === Police: Investigate highest Mafia suspect ===
        elif current_role == config.ROLE_POLICE:
            # Exclude already investigated players
            candidates = [
                pid for pid in alive_ids
                if pid != self.id and pid not in self.investigated_players
            ]
            
            if not candidates:
                # All investigated, pick highest suspect
                candidates = [pid for pid in alive_ids if pid != self.id]
                if not candidates:
                    return -1
            
            # Investigate highest Mafia suspect
            mafia_scores = np.array([
                self.belief[pid, config.ROLE_MAFIA] for pid in candidates
            ])
            
            target = self._select_target_softmax(candidates, mafia_scores, temperature=0.1)
            self.investigated_players.add(target)
            return target
        
        return -1
