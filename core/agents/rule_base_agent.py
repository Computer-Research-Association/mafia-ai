import random
from typing import List, Dict, Optional
import numpy as np
from itertools import combinations

from core.agents.base_agent import BaseAgent
from core.engine.state import GameStatus, GameAction
from config import EventType, Role, Phase

# Import the Bayesian Inference components
from core.agents.rba.engine import BayesianInferenceEngine
from core.agents.rba.constants import (
    RBAEventType, ABSTAIN_ENTROPY_THRESHOLD, 
    INITIAL_VOTE_TEMPERATURE, MIN_VOTE_TEMPERATURE, VOTE_TEMP_DECAY_RATE
)
from core.agents.rba.strategies import calculate_entropy, should_abstain, softmax_selection

# [Tuning] 시민 단합력을 위한 상수 조정
BASE_SUSPICION = 10.0
BASE_ATTENTION = 10.0

class RuleBaseAgent(BaseAgent):
    def __init__(self, player_id: int, role: Role):
        super().__init__(player_id, role)
        self.reset()

    def reset(self):
        self.suspicion = BASE_SUSPICION
        self.attention = BASE_ATTENTION
        self.bayesian_engine: Optional[BayesianInferenceEngine] = None
        self.public_bayesian_engine: Optional[BayesianInferenceEngine] = None
        
        self.known_mafia: List[int] = []
        self.known_safe: List[int] = []
        self.investigated_log: List[int] = []
        
        self.claimed_role: Optional[Role] = None
        self.is_hiding_role = False
        self.declared_self_heal = False
        self.is_confirmed = False
        self.is_proven_citizen = False
        self.protected_target = -1
        
        self.last_processed_event_idx = 0
        self.daily_votes = {}
        
        if self.role == Role.POLICE and random.random() < 0.8: self.is_hiding_role = True
        elif self.role == Role.DOCTOR and random.random() < 0.5: self.is_hiding_role = True
            
        self.public_claims = []
        self.last_night_healed_id = -1
        self.player_stats = {}

    def get_action(self, status: GameStatus) -> GameAction:
        self.role = status.my_role
        self.my_id = self.id
        self._sync_state(status)

        alive_others = [p.id for p in status.players if p.alive and p.id != self.id]
        if not alive_others: return GameAction(target_id=-1)

        if status.phase == Phase.DAY_DISCUSSION: return self._act_discussion(status)
        elif status.phase == Phase.DAY_VOTE: return self._act_vote(status)
        elif status.phase == Phase.DAY_EXECUTE: return self._act_execute(status)
        elif status.phase == Phase.NIGHT: return self._act_night(status, alive_others)
        
        return GameAction(target_id=-1)

    def _sync_state(self, status: GameStatus):
        num_players = len(status.players)
        mafia_count = 2
        
        if self.bayesian_engine is None:
            mafia_team_ids = []
            if self.role == Role.MAFIA:
                mafia_team_ids.append(self.id)
                for event in status.action_history:
                    if event.event_type == EventType.POLICE_RESULT and event.actor_id == -1 and event.value == Role.MAFIA:
                        if event.target_id not in mafia_team_ids:
                            mafia_team_ids.append(event.target_id)
            
            self.bayesian_engine = BayesianInferenceEngine(
                num_players=num_players, agent_id=self.id, agent_team="mafia" if self.role == Role.MAFIA else "citizen",
                mafia_count=mafia_count, mafia_team_ids=mafia_team_ids
            )
            if self.role == Role.MAFIA:
                for member_id in mafia_team_ids:
                    if member_id not in self.known_mafia:
                        self.known_mafia.append(member_id)
            self.public_bayesian_engine = BayesianInferenceEngine(
                num_players=num_players, agent_id=self.id, agent_team='citizen',
                mafia_count=mafia_count, mafia_team_ids=[]
            )

        if not self.player_stats:
            for p in status.players:
                self.player_stats[p.id] = { "claimed_role": None, "is_confirmed": False }

        new_events = status.action_history[self.last_processed_event_idx:]
        days_with_new_votes = set()

        for event in new_events:
            actor, target = event.actor_id, event.target_id
            
            if event.event_type == EventType.EXECUTE and event.value:
                is_mafia = (event.value == Role.MAFIA)
                self.bayesian_engine.set_player_probability(target, is_mafia=is_mafia)
                self.public_bayesian_engine.set_player_probability(target, is_mafia=is_mafia)
                
                if is_mafia:
                    if target not in self.known_mafia: self.known_mafia.append(target)
                else:
                    if target not in self.known_safe: self.known_safe.append(target)

                claims_against_target = [c for c in self.public_claims if c["targetId"] == target]
                for claim in claims_against_target:
                    event_type = RBAEventType.ACCUSATION_WAS_RIGHT if is_mafia else RBAEventType.ACCUSATION_WAS_WRONG
                    self.bayesian_engine.update(event_type, actor_id=claim["claimerId"], status=status)
                    self.public_bayesian_engine.update(event_type, actor_id=claim["claimerId"], status=status)

            if event.event_type == EventType.VOTE and actor is not None and target is not None:
                days_with_new_votes.add(event.day)
                self.daily_votes.setdefault(event.day, {})[actor] = target
                
                self.bayesian_engine.update(RBAEventType.PLAYER_VOTED, actor_id=actor, status=status)
                self.public_bayesian_engine.update(RBAEventType.PLAYER_VOTED, actor_id=actor, status=status)
                self.bayesian_engine.update(RBAEventType.PLAYER_WAS_VOTED_FOR, actor_id=target, status=status)
                self.public_bayesian_engine.update(RBAEventType.PLAYER_WAS_VOTED_FOR, actor_id=target, status=status)

            if event.event_type == EventType.CLAIM and event.value is not None:
                if actor is None or actor not in self.player_stats: continue

                # Accusation claim ("Player X is Mafia")
                if target is not None and target != actor:
                    self.public_claims.append({"claimerId": actor, "targetId": target, "day": event.day})
                    self.bayesian_engine.update(RBAEventType.PLAYER_ACCUSED, actor_id=target, status=status)
                    self.public_bayesian_engine.update(RBAEventType.PLAYER_ACCUSED, actor_id=target, status=status)
                
                # Self-claim ("I am Role Y")
                else:
                    claimed_role = event.value
                    claimer_id = actor

                    # Check for counter-claims on unique roles
                    existing_claimant_id = None
                    if claimed_role in [Role.POLICE, Role.DOCTOR]:
                        for p_id, stats in self.player_stats.items():
                            if p_id != claimer_id and stats.get("claimed_role") == claimed_role:
                                existing_claimant_id = p_id
                                break
                    
                    # Update this player's claimed role *after* checking for contradiction
                    self.player_stats[claimer_id]["claimed_role"] = claimed_role

                    if existing_claimant_id is not None:
                        # Counter-claim detected! Increase suspicion for both.
                        self.bayesian_engine.update(RBAEventType.ROLE_COUNTERCLAIMED, actor_id=claimer_id, status=status)
                        self.public_bayesian_engine.update(RBAEventType.ROLE_COUNTERCLAIMED, actor_id=claimer_id, status=status)
                        self.bayesian_engine.update(RBAEventType.ROLE_COUNTERCLAIMED, actor_id=existing_claimant_id, status=status)
                        self.public_bayesian_engine.update(RBAEventType.ROLE_COUNTERCLAIMED, actor_id=existing_claimant_id, status=status)
                    else:
                        # First time this role is claimed (or it's a non-unique role)
                        self.bayesian_engine.update(RBAEventType.ROLE_CLAIMED, actor_id=claimer_id, status=status)
                        self.public_bayesian_engine.update(RBAEventType.ROLE_CLAIMED, actor_id=claimer_id, status=status)

            if event.event_type == EventType.POLICE_RESULT and event.actor_id == self.id:
                is_mafia = (event.value == Role.MAFIA)
                self.bayesian_engine.set_player_probability(target, is_mafia=is_mafia)
                if is_mafia:
                    if target not in self.known_mafia: self.known_mafia.append(target)
                else:
                    if target not in self.known_safe: self.known_safe.append(target)

        # --- Correlation Analysis from Voting Patterns ---
        for day in days_with_new_votes:
            voters_by_target = {}
            for voter, target in self.daily_votes.get(day, {}).items():
                if target is not None:
                    voters_by_target.setdefault(target, []).append(voter)
            
            for target, voters in voters_by_target.items():
                if len(voters) > 1:
                    for p1, p2 in combinations(voters, 2):
                        strength = 0.05
                        self.bayesian_engine.update_correlation(p1, p2, strength)
                        self.public_bayesian_engine.update_correlation(p1, p2, strength)

        for p_id in self.known_mafia:
            self.bayesian_engine.set_player_probability(p_id, is_mafia=True)
            self.public_bayesian_engine.set_player_probability(p_id, is_mafia=True)
        for p_id in self.known_safe:
            self.bayesian_engine.set_player_probability(p_id, is_mafia=False)
            self.public_bayesian_engine.set_player_probability(p_id, is_mafia=False)
            
        self.last_processed_event_idx = len(status.action_history)

    def _act_discussion(self, status: GameStatus) -> GameAction:
        survivors = [p for p in status.players if p.alive and p.id != self.id]
        if not survivors:
            return GameAction(target_id=-1)

        probabilities = self.bayesian_engine.get_mafia_probabilities()
        public_probabilities = self.public_bayesian_engine.get_mafia_probabilities()
        my_public_mafia_prob = public_probabilities[self.id]

        if not self.claimed_role and my_public_mafia_prob > 0.6:
            if self.role == Role.MAFIA:
                gamble = random.random()
                if gamble < 0.5: return self._do_claim(Role.POLICE)
                elif gamble < 0.85: return self._do_claim(Role.DOCTOR)
                else: return self._do_claim(Role.CITIZEN)
            else:
                self.is_hiding_role = False
                return self._do_claim(self.role)

        personal_target, max_personal_sus = -1, -1
        for p in survivors:
            if self.role == Role.MAFIA and p.id in self.known_mafia: continue
            if probabilities[p.id] > max_personal_sus:
                max_personal_sus, personal_target = probabilities[p.id], p.id
        
        if personal_target != -1 and max_personal_sus > 0.5:
             if self.role == Role.MAFIA or random.random() < 0.5:
                 return GameAction(target_id=personal_target, claim_role=Role.MAFIA)

        if self.role == Role.POLICE:
            counter = [p for p in survivors if self.player_stats.get(p.id, {}).get("claimed_role") == Role.POLICE]
            if counter:
                self.is_hiding_role = False
                return GameAction(target_id=counter[0].id, claim_role=Role.MAFIA)
            
            found_mafia = [m for m in self.known_mafia if m in [p.id for p in survivors]]
            if found_mafia:
                self.is_hiding_role = False
                return GameAction(target_id=found_mafia[0], claim_role=Role.MAFIA)
            
            if not self.is_hiding_role and not self.claimed_role:
                return self._do_claim(Role.POLICE)

        elif self.role == Role.DOCTOR:
            counter = [p for p in survivors if self.player_stats.get(p.id, {}).get("claimed_role") == Role.DOCTOR]
            if counter and not self.claimed_role:
                return self._do_claim(Role.DOCTOR)
            if not self.is_hiding_role and not self.claimed_role:
                 return self._do_claim(Role.DOCTOR)
        
        return GameAction(target_id=-1)

    def _act_vote(self, status: GameStatus) -> GameAction:
        if self.bayesian_engine and len(self.known_mafia) >= self.bayesian_engine.initial_mafia_count:
            return GameAction(target_id=-1)

        alive_players = [p for p in status.players if p.alive]
        if not alive_players or len(alive_players) <= 2: return GameAction(target_id=-1)

        # --- Temperature Annealing ---
        total_players = len(status.players)
        progress = (total_players - len(alive_players)) / (total_players - 2) if (total_players - 2) > 0 else 1
        progress = np.clip(progress, 0, 1)
        
        current_temp = MIN_VOTE_TEMPERATURE + (INITIAL_VOTE_TEMPERATURE - MIN_VOTE_TEMPERATURE) * np.exp(-progress * VOTE_TEMP_DECAY_RATE)

        # --- Candidate Selection ---
        probabilities = self.bayesian_engine.get_mafia_probabilities()
        mask = np.ones_like(probabilities)
        mask[self.id] = 0
        
        alive_player_ids = [p.id for p in alive_players]
        for i in range(len(probabilities)):
            if i not in alive_player_ids: mask[i] = 0
        for safe_id in self.known_safe:
            if safe_id in alive_player_ids: mask[safe_id] = 0
        if self.role == Role.MAFIA:
            for mafia_id in self.known_mafia:
                if mafia_id in alive_player_ids: mask[mafia_id] = 0
        
        masked_probabilities = probabilities * mask
        if np.sum(masked_probabilities) == 0: return GameAction(target_id=-1)
        normalized_probs = masked_probabilities / np.sum(masked_probabilities)

        # --- Decision ---
        entropy = calculate_entropy(normalized_probs)
        if should_abstain(entropy, ABSTAIN_ENTROPY_THRESHOLD): return GameAction(target_id=-1)
        
        target_id = softmax_selection(normalized_probs, temperature=current_temp)
        return GameAction(target_id=target_id)

    def _act_execute(self, status: GameStatus) -> GameAction:
        votes = {}
        for e in status.action_history:
            if e.day == status.day and e.phase == Phase.DAY_VOTE and e.event_type == EventType.VOTE:
                 if e.target_id is not None and e.target_id != -1:
                    votes[e.target_id] = votes.get(e.target_id, 0) + 1
        
        if not votes: return GameAction(target_id=-1)
        target_id = max(votes, key=votes.get)
        if target_id == self.id: return GameAction(target_id=-1)
        if self.role == Role.MAFIA and target_id in self.known_mafia: return GameAction(target_id=-1)
        if target_id in self.known_safe: return GameAction(target_id=-1)

        mafia_prob = self.bayesian_engine.get_mafia_probabilities()[target_id]
        return GameAction(target_id=target_id) if mafia_prob > 0.5 else GameAction(target_id=-1)

    def _act_night(self, status: GameStatus, targets: List[int]) -> GameAction:
        alive_player_ids = [p.id for p in status.players if p.alive]
        probabilities = self.bayesian_engine.get_mafia_probabilities()

        if self.role == Role.MAFIA:
            candidates = {p_id: 1.0 for p_id in alive_player_ids if p_id != self.id and p_id not in self.known_mafia}
            if not candidates: return GameAction(target_id=-1)
            for p_id in candidates:
                candidates[p_id] = 1.0 - probabilities[p_id]
                claimed_role = self.player_stats.get(p_id, {}).get("claimed_role")
                if claimed_role in [Role.POLICE, Role.DOCTOR]: candidates[p_id] *= 3.0
            candidate_ids, weights = list(candidates.keys()), np.array(list(candidates.values()))
            if np.sum(weights) <= 0: target_id = np.random.choice(candidate_ids)
            else: target_id = candidate_ids[softmax_selection(weights, temperature=0.5)]
            return GameAction(target_id=target_id)
            
        elif self.role == Role.DOCTOR:
            candidates = {p_id: 1.0 for p_id in alive_player_ids}
            for p_id in candidates:
                candidates[p_id] = 1.0 - probabilities[p_id]
            for p_id, st in self.player_stats.items():
                if st.get("claimed_role") == Role.POLICE and p_id in candidates: candidates[p_id] *= 2.0
            
            uncertain_indices = [p.id for p in status.players if p.alive and 0 < probabilities[p.id] < 1]
            entropy = calculate_entropy(probabilities[uncertain_indices]/np.sum(probabilities[uncertain_indices])) if uncertain_indices and np.sum(probabilities[uncertain_indices]) > 0 else 0
            self_heal_boost = 1.0 + (entropy / (ABSTAIN_ENTROPY_THRESHOLD or 1.0))
            if self.id in candidates: candidates[self.id] *= self_heal_boost

            candidate_ids, weights = list(candidates.keys()), np.array(list(candidates.values()))
            if np.sum(weights) <= 0: return GameAction(target_id=self.id)
            target_id = candidate_ids[softmax_selection(weights, temperature=0.5)]
            return GameAction(target_id=target_id)

        elif self.role == Role.POLICE:
            candidates = [p_id for p_id in alive_player_ids if p_id != self.id and p_id not in self.investigated_log]
            if not candidates:
                self.investigated_log = []
                candidates = [p_id for p_id in alive_player_ids if p_id != self.id]
                if not candidates: return GameAction(target_id=-1)

            uncertainty = {}
            for p_id in candidates:
                p = np.clip(probabilities[p_id], 1e-9, 1 - 1e-9)
                uncertainty[p_id] = -p * np.log2(p) - (1-p) * np.log2(1-p)
            
            target_id = max(uncertainty, key=uncertainty.get) if uncertainty else (random.choice(candidates) if candidates else -1)
            if target_id != -1: self.investigated_log.append(target_id)
            return GameAction(target_id=target_id)
        
        return GameAction(target_id=-1)

    def _do_claim(self, role: Role) -> GameAction:
        self.claimed_role = role
        return GameAction(target_id=self.id, claim_role=role)