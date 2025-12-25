"""
Rational Rule-Based Agent (RBA) for Mafia Game

This module implements a unified rational agent that makes decisions based purely
on logical deduction and belief scores, without personality-based randomness.
"""

from typing import List, Set, Dict, Tuple
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
        
        self.aggressiveness = 0.5
        self.trust_threshold = 0.8
        
        self.memory = []
        self.trust_scores = {i: 0.5 for i in range(config.PLAYER_COUNT)}
        self.trust_scores[self.id] = 1.0
        
        self.role_claims = {}
        self.committed_target = -1
        
        self.accusation_history = []
        self.execution_history = []
        self.healed_players = set()
        self.no_death_nights = []
        
        initial_belief = 25.0
        self.belief = np.full((config.PLAYER_COUNT, 4), initial_belief, dtype=np.float32)
        
        self.belief[self.id] = 0.0
        self.belief[self.id, self.role] = 100.0
    
    def update_belief(self, game_status: dict):
        """Update belief matrix based on observed game events."""
        if not self.alive:
            return
        
        claims = game_status.get('claims', [])
        alive_players = game_status.get('alive_players', [])
        execution_result = game_status.get('execution_result', None)
        night_result = game_status.get('night_result', None)
        current_day = game_status.get('day', 1)
        
        for claim in claims:
            speaker_id = claim.get('speaker_id')
            claim_type = claim.get('type', 'NO_ACTION')
            reveal_role = claim.get('reveal_role', -1)
            target_id = claim.get('target_id', -1)
            assertion = claim.get('assertion', 'SUSPECT')
            
            if speaker_id == self.id or claim_type == 'NO_ACTION':
                continue
            
            self.memory.append((current_day, speaker_id, reveal_role, target_id, assertion))
            
            if reveal_role != -1:
                self._handle_role_claim(speaker_id, reveal_role)
            
            if target_id != -1:
                self._handle_assertion(speaker_id, target_id, assertion, reveal_role, current_day)
        
        if execution_result:
            self._process_execution(execution_result, alive_players, current_day)
        
        if night_result and self.role == config.ROLE_DOCTOR:
            self._process_doctor_heal(night_result, game_status)
        
        self._normalize_beliefs(alive_players)
    
    def _handle_role_claim(self, speaker_id: int, reveal_role: int):
        """Handle role claims and detect conflicts."""
        if reveal_role in self.role_claims.values():
            for other_id, other_role in self.role_claims.items():
                if other_role == reveal_role and other_id != speaker_id:
                    self.trust_scores[speaker_id] = 0.3
                    self.trust_scores[other_id] = 0.3
                    self._apply_belief_update(speaker_id, config.ROLE_MAFIA, 30.0, "role_conflict")
                    self._apply_belief_update(other_id, config.ROLE_MAFIA, 30.0, "role_conflict")
        
        self.role_claims[speaker_id] = reveal_role
        
        if reveal_role in [config.ROLE_POLICE, config.ROLE_DOCTOR]:
            self.trust_scores[speaker_id] = min(1.0, self.trust_scores.get(speaker_id, 0.5) + 0.2)
    
    def _handle_assertion(self, speaker_id: int, target_id: int, assertion: str, 
                         reveal_role: int, current_day: int):
        """Handle assertions about other players."""
        self.accusation_history.append((speaker_id, target_id, current_day))
        
        if target_id == self.id:
            self._handle_assertion_about_me(speaker_id, assertion, reveal_role)
            return
        
        trust_factor = self.trust_scores.get(speaker_id, 0.5)
        
        if assertion == "CONFIRMED_MAFIA":
            self._apply_belief_update(target_id, config.ROLE_MAFIA, 40.0 * trust_factor, "confirmed_mafia_claim")
        elif assertion == "CONFIRMED_CITIZEN":
            self._apply_belief_update(target_id, config.ROLE_MAFIA, -30.0 * trust_factor, "confirmed_citizen_claim")
        elif assertion == "SUSPECT":
            self._apply_belief_update(target_id, config.ROLE_MAFIA, 15.0 * trust_factor, "suspect_claim")
    
    def _handle_assertion_about_me(self, speaker_id: int, assertion: str, reveal_role: int):
        """Handle assertions targeting myself."""
        if assertion == "CONFIRMED_CITIZEN" and self.role != config.ROLE_MAFIA:
            if reveal_role == config.ROLE_POLICE:
                self.trust_scores[speaker_id] = 1.0
                self._apply_belief_update(speaker_id, config.ROLE_POLICE, 80.0, "self_verification")
                self._apply_belief_update(speaker_id, config.ROLE_MAFIA, -100.0, "self_verification")
        elif self.role != config.ROLE_MAFIA:
            self._apply_belief_update(speaker_id, config.ROLE_MAFIA, 15.0, "false_accusation")
            self.trust_scores[speaker_id] = max(0.0, self.trust_scores.get(speaker_id, 0.5) - 0.2)
        elif self.role == config.ROLE_MAFIA:
            self._apply_belief_update(speaker_id, config.ROLE_POLICE, 20.0, "accused_me_as_mafia")
    
    def _process_execution(self, execution_result: Tuple, alive_players: List[int], current_day: int):
        """Process execution results and update beliefs."""
        executed_id, team_alignment, day = execution_result
        
        self.execution_history.append((executed_id, team_alignment, day))
        self.belief[executed_id] = 0.0
        
        if team_alignment == "CITIZEN":
            self.belief[executed_id, config.ROLE_CITIZEN] = 100.0
            self._penalize_false_accusers(executed_id, day, alive_players)
            self._penalize_counter_claimers(executed_id, alive_players)
        elif team_alignment == "MAFIA":
            self.belief[executed_id, config.ROLE_MAFIA] = 100.0
            self._reward_correct_accusers(executed_id, day)
    
    def _penalize_false_accusers(self, executed_id: int, day: int, alive_players: List[int]):
        """Penalize players who falsely accused an innocent player."""
        false_accusers = set()
        
        for accuser_id, accused_id, acc_day in self.accusation_history:
            if accused_id == executed_id and acc_day == day:
                false_accusers.add(accuser_id)
        
        for mem_day, speaker_id, claimed_role, target_id, assertion in self.memory:
            if target_id == executed_id and mem_day == day:
                if assertion in ["CONFIRMED_MAFIA", "SUSPECT"]:
                    false_accusers.add(speaker_id)
        
        for accuser_id in false_accusers:
            if accuser_id != self.id and accuser_id in alive_players:
                self.trust_scores[accuser_id] = 0.0
                self._apply_belief_update(accuser_id, config.ROLE_MAFIA, 50.0, 
                                        f"false_accuser_{executed_id}")
    
    def _penalize_counter_claimers(self, executed_id: int, alive_players: List[int]):
        """Penalize players who counter-claimed the executed player's role."""
        if executed_id not in self.role_claims:
            return
        
        claimed_role = self.role_claims[executed_id]
        for other_id, other_role in self.role_claims.items():
            if other_role == claimed_role and other_id != executed_id and other_id in alive_players:
                self.trust_scores[other_id] = 0.0
                self._apply_belief_update(other_id, config.ROLE_MAFIA, 80.0, "false_counter_claim")
    
    def _reward_correct_accusers(self, executed_id: int, day: int):
        """Reward players who correctly accused a mafia member."""
        for accuser_id, accused_id, acc_day in self.accusation_history:
            if accused_id == executed_id and acc_day == day:
                self.trust_scores[accuser_id] = min(1.0, self.trust_scores.get(accuser_id, 0.5) + 0.4)
                if self.role != config.ROLE_MAFIA:
                    self._apply_belief_update(accuser_id, config.ROLE_POLICE, 20.0, f"found_mafia_{executed_id}")
                    self._apply_belief_update(accuser_id, config.ROLE_MAFIA, -30.0, f"found_mafia_{executed_id}")
    
    def _process_doctor_heal(self, night_result: Dict, game_status: dict):
        """Process doctor's heal results."""
        no_death = night_result.get('no_death', False)
        last_healed = night_result.get('last_healed', -1)
        
        if no_death and last_healed != -1:
            self.healed_players.add(last_healed)
            self.no_death_nights.append(game_status.get('day', 1))
            
            self._apply_belief_update(last_healed, config.ROLE_CITIZEN, 15.0, "successful_heal")
            self._apply_belief_update(last_healed, config.ROLE_MAFIA, -20.0, "successful_heal")
    
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
        """Normalize beliefs to maintain consistency."""
        for player_id in range(config.PLAYER_COUNT):
            if player_id == self.id or player_id not in alive_players:
                continue
            
            total = np.sum(self.belief[player_id])
            if total > 200:
                self.belief[player_id] *= (200 / total)
            elif total < -200:
                self.belief[player_id] *= (-200 / total)
    
    def _is_police_dead(self) -> bool:
        """Check if Police is confirmed dead."""
        for executed_id, team_alignment, day in self.execution_history:
            if executed_id in self.role_claims:
                if self.role_claims[executed_id] == config.ROLE_POLICE:
                    return True
            if self.belief[executed_id, config.ROLE_POLICE] > 70.0:
                return True
        return False
    
    def _get_alive_count(self, players: List["BaseCharacter"]) -> int:
        """Count alive players."""
        return sum(1 for p in players if p.alive)
    
    def decide_claim(self, players: List["BaseCharacter"], current_day: int = 1) -> dict:
        """Returns a dictionary with structured claim details."""
        alive_ids = self._get_alive_ids(players, exclude_me=True)
        if not alive_ids:
            self.committed_target = -1
            return {"type": "NO_ACTION", "reveal_role": -1, "target_id": -1, "assertion": "SUSPECT"}
        
        self.committed_target = -1
        
        counter_claim = self._check_counter_claim(alive_ids)
        if counter_claim:
            return counter_claim
        
        if self.role == config.ROLE_POLICE:
            return self._police_claim_strategy(players, alive_ids, current_day)
        elif self.role == config.ROLE_DOCTOR:
            return self._doctor_claim_strategy(players, alive_ids, current_day)
        elif self.role == config.ROLE_CITIZEN:
            return self._citizen_claim_strategy(alive_ids, current_day)
        elif self.role == config.ROLE_MAFIA:
            return self._mafia_claim_strategy(players, alive_ids, current_day)
        
        return {"type": "NO_ACTION", "reveal_role": -1, "target_id": -1, "assertion": "SUSPECT"}
    
    def _check_counter_claim(self, alive_ids: List[int]) -> Dict:
        """Check if someone claimed my role and return counter-claim if needed."""
        for pid, claimed_role in self.role_claims.items():
            if claimed_role == self.role and pid != self.id and pid in alive_ids:
                self.committed_target = pid
                return {
                    "type": "CLAIM",
                    "reveal_role": self.role,
                    "target_id": pid,
                    "assertion": "CONFIRMED_MAFIA"
                }
        return None
    
    def _police_claim_strategy(self, players: List["BaseCharacter"], 
                              alive_ids: List[int], current_day: int) -> Dict:
        """Police claim strategy."""
        alive_count = self._get_alive_count(players)
        
        if alive_count <= 4:
            for pid in alive_ids:
                if pid in self.investigated_players and self.belief[pid, config.ROLE_MAFIA] < -50:
                    return {
                        "type": "CLAIM",
                        "reveal_role": config.ROLE_POLICE,
                        "target_id": pid,
                        "assertion": "CONFIRMED_CITIZEN"
                    }
        
        for mafia_id in self.confirmed_mafia:
            if mafia_id in alive_ids:
                self.should_reveal = True
                self.committed_target = mafia_id
                return {
                    "type": "CLAIM",
                    "reveal_role": config.ROLE_POLICE,
                    "target_id": mafia_id,
                    "assertion": "CONFIRMED_MAFIA"
                }
        
        if current_day <= 2:
            return {"type": "NO_ACTION", "reveal_role": -1, "target_id": -1, "assertion": "SUSPECT"}
        
        return self._claim_with_silence_threshold(alive_ids)
    
    def _doctor_claim_strategy(self, players: List["BaseCharacter"], 
                              alive_ids: List[int], current_day: int) -> Dict:
        """Doctor claim strategy."""
        alive_count = self._get_alive_count(players)
        police_is_dead = self._is_police_dead()
        
        if police_is_dead or alive_count <= 3:
            best_citizen = self._find_best_citizen_to_vouch(alive_ids)
            if best_citizen is not None:
                return {
                    "type": "CLAIM",
                    "reveal_role": config.ROLE_DOCTOR,
                    "target_id": best_citizen,
                    "assertion": "CONFIRMED_CITIZEN"
                }
        
        for healed_pid in self.healed_players:
            if healed_pid in alive_ids:
                accusations_against = sum(1 for _, target, _ in self.accusation_history if target == healed_pid)
                if accusations_against >= 2 and current_day >= 3:
                    return {
                        "type": "CLAIM",
                        "reveal_role": config.ROLE_DOCTOR,
                        "target_id": healed_pid,
                        "assertion": "CONFIRMED_CITIZEN"
                    }
        
        if current_day <= 2:
            return {"type": "NO_ACTION", "reveal_role": -1, "target_id": -1, "assertion": "SUSPECT"}
        
        return self._claim_with_silence_threshold(alive_ids)
    
    def _find_best_citizen_to_vouch(self, alive_ids: List[int]) -> int:
        """Find the best citizen to vouch for."""
        if self.healed_players:
            for healed_pid in self.healed_players:
                if healed_pid in alive_ids:
                    return healed_pid
        
        citizen_scores = [(pid, -self.belief[pid, config.ROLE_MAFIA]) for pid in alive_ids]
        citizen_scores.sort(key=lambda x: x[1], reverse=True)
        return citizen_scores[0][0] if citizen_scores else None
    
    def _claim_with_silence_threshold(self, alive_ids: List[int]) -> Dict:
        """Make a claim only if suspicion exceeds silence threshold."""
        mafia_scores = [(pid, self.belief[pid, config.ROLE_MAFIA]) for pid in alive_ids]
        mafia_scores.sort(key=lambda x: x[1], reverse=True)
        
        silence_threshold = 40.0 * self.aggressiveness
        if mafia_scores[0][1] > silence_threshold:
            self.committed_target = mafia_scores[0][0]
            return {
                "type": "CLAIM",
                "reveal_role": -1,
                "target_id": mafia_scores[0][0],
                "assertion": "SUSPECT"
            }
        
        return {"type": "NO_ACTION", "reveal_role": -1, "target_id": -1, "assertion": "SUSPECT"}
        return {"type": "NO_ACTION", "reveal_role": -1, "target_id": -1, "assertion": "SUSPECT"}
    
    def _citizen_claim_strategy(self, alive_ids: List[int], current_day: int) -> Dict:
        """Citizen claim strategy."""
        if current_day <= 2:
            return {"type": "NO_ACTION", "reveal_role": -1, "target_id": -1, "assertion": "SUSPECT"}
        
        return self._claim_with_silence_threshold(alive_ids)
    
    def _mafia_claim_strategy(self, players: List["BaseCharacter"], 
                             alive_ids: List[int], current_day: int) -> Dict:
        """Mafia claim strategy."""
        non_mafia_alive = [pid for pid in alive_ids if players[pid].role != config.ROLE_MAFIA]
        
        if not non_mafia_alive or current_day == 1:
            return {"type": "NO_ACTION", "reveal_role": -1, "target_id": -1, "assertion": "SUSPECT"}
        
        police_suspects = [(pid, self.belief[pid, config.ROLE_POLICE]) for pid in non_mafia_alive]
        police_suspects.sort(key=lambda x: x[1], reverse=True)
        
        threshold = 60.0 * self.aggressiveness
        if police_suspects[0][1] > threshold:
            self.committed_target = police_suspects[0][0]
            return {
                "type": "CLAIM",
                "reveal_role": -1,
                "target_id": police_suspects[0][0],
                "assertion": "SUSPECT"
            }
        
        if np.random.random() < (0.4 * self.aggressiveness):
            target = np.random.choice(non_mafia_alive)
            self.committed_target = target
            return {
                "type": "CLAIM",
                "reveal_role": -1,
                "target_id": target,
                "assertion": "SUSPECT"
            }
        
        return {"type": "NO_ACTION", "reveal_role": -1, "target_id": -1, "assertion": "SUSPECT"}
    
    def decide_vote(self, players: List["BaseCharacter"], current_day: int = 1) -> int:
        """Deterministic voting using argmax strategy."""
        alive_ids = self._get_alive_ids(players, exclude_me=True)
        if not alive_ids:
            return -1
        
        if self.committed_target != -1 and self.committed_target in alive_ids:
            vote_target = self.committed_target
            self.committed_target = -1  # Reset after voting
            return vote_target
        
        self.committed_target = -1
        
        if self.role != config.ROLE_MAFIA:
            return self._citizen_vote_strategy(players, alive_ids, current_day)
        else:
            return self._mafia_vote_strategy(players, alive_ids)
    
    def _citizen_vote_strategy(self, players: List["BaseCharacter"], 
                              alive_ids: List[int], current_day: int) -> int:
        """Citizen team voting strategy."""
        mafia_scores = [(pid, self.belief[pid, config.ROLE_MAFIA]) for pid in alive_ids]
        mafia_scores.sort(key=lambda x: (-x[1], x[0]))
        
        highest_score = mafia_scores[0][1]
        alive_count = self._get_alive_count(players)
        
        if alive_count <= 4:
            if highest_score < 0:
                unknown_players = self._find_unknown_players(alive_ids)
                if unknown_players:
                    unknown_players.sort()
                    return unknown_players[0]
            return mafia_scores[0][0]
        
        threshold = 60.0 * self.trust_threshold if current_day <= 2 else 40.0 * self.trust_threshold
        
        if highest_score > threshold:
            return mafia_scores[0][0]
        
        return -1
    
    def _find_unknown_players(self, alive_ids: List[int]) -> List[int]:
        """Find players not confirmed as citizens."""
        unknown_players = []
        for pid in alive_ids:
            is_confirmed = False
            for mem_day, speaker_id, claimed_role, target_id, assertion in self.memory:
                if target_id == pid and assertion == "CONFIRMED_CITIZEN":
                    trust = self.trust_scores.get(speaker_id, 0.5)
                    if trust > 0.6:
                        is_confirmed = True
                        break
            
            if not is_confirmed:
                unknown_players.append(pid)
        
        return unknown_players
    
    def _mafia_vote_strategy(self, players: List["BaseCharacter"], alive_ids: List[int]) -> int:
        """Mafia voting strategy."""
        non_mafia_alive = [pid for pid in alive_ids if players[pid].role != config.ROLE_MAFIA]
        
        if not non_mafia_alive:
            return -1
        
        threat_scores = [
            (pid, 
             self.belief[pid, config.ROLE_POLICE] * 2.0 +
             self.belief[pid, config.ROLE_DOCTOR] * 1.5)
            for pid in non_mafia_alive
        ]
        
        threat_scores.sort(key=lambda x: (-x[1], x[0]))
        return threat_scores[0][0]
    
    def decide_night_action(self, players: List["BaseCharacter"], current_role: int) -> int:
        """Decide night action based on role."""
        alive_ids = self._get_alive_ids(players, exclude_me=False)
        if not alive_ids:
            return -1
        
        if current_role == config.ROLE_MAFIA:
            return self._mafia_night_action(players, alive_ids)
        elif current_role == config.ROLE_DOCTOR:
            return self._doctor_night_action(players, alive_ids)
        elif current_role == config.ROLE_POLICE:
            return self._police_night_action(alive_ids)
        
        return -1
    
    def _mafia_night_action(self, players: List["BaseCharacter"], alive_ids: List[int]) -> int:
        """Mafia night kill strategy."""
        candidates = [
            pid for pid in alive_ids
            if pid != self.id and players[pid].role != config.ROLE_MAFIA
        ]
        
        if not candidates:
            return -1
        
        threat_scores = np.array([
            (100.0 if players[pid].should_reveal and players[pid].role == config.ROLE_POLICE else 0.0) +
            self.belief[pid, config.ROLE_POLICE] * 2.0 +
            self.belief[pid, config.ROLE_DOCTOR] * 1.2
            for pid in candidates
        ])
        
        return self._select_target_softmax(candidates, threat_scores, temperature=0.1)
    
    def _doctor_night_action(self, players: List["BaseCharacter"], alive_ids: List[int]) -> int:
        """Doctor night protection strategy."""
        candidates = alive_ids
        
        if not candidates:
            return -1
        
        protection_scores = np.array([
            (100.0 if players[pid].should_reveal and players[pid].role == config.ROLE_POLICE else 0.0) +
            self.belief[pid, config.ROLE_POLICE] * 1.5 +
            (10.0 if pid == self.id else 0.0)
            for pid in candidates
        ])
        
        return self._select_target_softmax(candidates, protection_scores, temperature=0.1)
    
    def _police_night_action(self, alive_ids: List[int]) -> int:
        """Police night investigation strategy."""
        candidates = [
            pid for pid in alive_ids
            if pid != self.id and pid not in self.investigated_players
        ]
        
        if not candidates:
            candidates = [pid for pid in alive_ids if pid != self.id]
            if not candidates:
                return -1
        
        mafia_scores = np.array([self.belief[pid, config.ROLE_MAFIA] for pid in candidates])
        return self._select_target_softmax(candidates, mafia_scores, temperature=0.1)
    
    def decide_night_action(self, players: List["BaseCharacter"], current_role: int) -> int:
        """Decide night action based on role."""
        alive_ids = self._get_alive_ids(players, exclude_me=False)
        if not alive_ids:
            return -1
        
        if current_role == config.ROLE_MAFIA:
            return self._mafia_night_action(players, alive_ids)
        elif current_role == config.ROLE_DOCTOR:
            return self._doctor_night_action(players, alive_ids)
        elif current_role == config.ROLE_POLICE:
            return self._police_night_action(alive_ids)
        
        return -1
    
    def _mafia_night_action(self, players: List["BaseCharacter"], alive_ids: List[int]) -> int:
        """Mafia night kill strategy."""
        candidates = [
            pid for pid in alive_ids
            if pid != self.id and players[pid].role != config.ROLE_MAFIA
        ]
        
        if not candidates:
            return -1
        
        threat_scores = np.array([
            (100.0 if players[pid].should_reveal and players[pid].role == config.ROLE_POLICE else 0.0) +
            self.belief[pid, config.ROLE_POLICE] * 2.0 +
            self.belief[pid, config.ROLE_DOCTOR] * 1.2
            for pid in candidates
        ])
        
        return self._select_target_softmax(candidates, threat_scores, temperature=0.1)
    
    def _doctor_night_action(self, players: List["BaseCharacter"], alive_ids: List[int]) -> int:
        """Doctor night protection strategy."""
        candidates = alive_ids
        
        if not candidates:
            return -1
        
        protection_scores = np.array([
            (100.0 if players[pid].should_reveal and players[pid].role == config.ROLE_POLICE else 0.0) +
            self.belief[pid, config.ROLE_POLICE] * 1.5 +
            (10.0 if pid == self.id else 0.0)
            for pid in candidates
        ])
        
        return self._select_target_softmax(candidates, protection_scores, temperature=0.1)
    
    def _police_night_action(self, alive_ids: List[int]) -> int:
        """Police night investigation strategy."""
        candidates = [
            pid for pid in alive_ids
            if pid != self.id and pid not in self.investigated_players
        ]
        
        if not candidates:
            candidates = [pid for pid in alive_ids if pid != self.id]
            if not candidates:
                return -1
        
        mafia_scores = np.array([self.belief[pid, config.ROLE_MAFIA] for pid in candidates])
        target = self._select_target_softmax(candidates, mafia_scores, temperature=0.1)
        self.investigated_players.add(target)
        return target
