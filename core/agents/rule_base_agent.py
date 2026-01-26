import random
from typing import List, Dict, Optional
import numpy as np

from core.agents.base_agent import BaseAgent
from core.engine.state import GameStatus, GameAction
from config import EventType, Role, Phase

# Import the Bayesian Inference components
from core.agents.rba.engine import BayesianInferenceEngine
from core.agents.rba.constants import RBAEventType, ABSTAIN_ENTROPY_THRESHOLD, VOTE_TEMPERATURE
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
        # The Bayesian engine replaces the old suspicion system
        self.bayesian_engine: Optional[BayesianInferenceEngine] = None
        
        self.known_mafia: List[int] = []
        self.known_safe: List[int] = []
        self.investigated_log: List[int] = []
        
        self.claimed_role: Optional[Role] = None
        self.is_hiding_role = False
        self.declared_self_heal = False
        self.is_confirmed = False
        self.is_proven_citizen = False
        self.protected_target = -1
        
        # Initialization
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
        # 1. Initialize Bayesian Engine if it's the first time
        if self.bayesian_engine is None:
            player_roles = [p.role.name.lower() if p.role else 'citizen' for p in status.players]
            self.bayesian_engine = BayesianInferenceEngine(
                num_players=num_players,
                agent_id=self.id,
                agent_team=self.role.to_team(),
                player_roles=player_roles,
                known_mafia_ids=self.known_mafia
            )

        # 2. Initialize player_stats if not present
        if not self.player_stats:
            for p in status.players:
                self.player_stats[p.id] = {
                    "claimed_role": None,
                    "is_confirmed": False,
                }

        # 3. Replay history to update Bayesian posteriors and other stats
        # Create a clean slate for stats that are rebuilt from history
        for p_id in self.player_stats:
            self.player_stats[p_id]["claimed_role"] = None
            self.player_stats[p_id]["is_confirmed"] = False
        self.public_claims = []
        
        # Re-initialize engine to replay history. This is inefficient but matches the original agent's structure.
        player_roles = [p.role.name.lower() if p.role else 'citizen' for p in status.players]
        self.bayesian_engine.reset_and_replay(status.action_history, player_roles, self.known_mafia)

        for event in status.action_history:
            actor = event.actor_id
            target = event.target_id
            
            # --- Handle Definitive Information ---
            # A player's role is revealed upon death
            if event.event_type == EventType.EXECUTE and event.value:
                is_mafia = (event.value == Role.MAFIA)
                self.bayesian_engine.set_player_probability(target, is_mafia=is_mafia)
                if is_mafia:
                    if target not in self.known_mafia: self.known_mafia.append(target)
                else:
                    if target not in self.known_safe: self.known_safe.append(target)

                # Check if past accusations against this player were right or wrong
                for claim in self.public_claims:
                    if claim["targetId"] == target:
                        event_type = RBAEventType.ACCUSATION_WAS_RIGHT if is_mafia else RBAEventType.ACCUSATION_WAS_WRONG
                        self.bayesian_engine.update(event_type, actor_id=claim["claimerId"])

            # A police investigation reveals a role
            if event.event_type == EventType.POLICE_RESULT and event.actor_id == self.id:
                is_mafia = (event.value == Role.MAFIA)
                self.bayesian_engine.set_player_probability(target, is_mafia=is_mafia)
                if is_mafia:
                    if target not in self.known_mafia: self.known_mafia.append(target)
                else:
                    if target not in self.known_safe: self.known_safe.append(target)

            # --- Handle Player Actions (Evidence) ---
            if event.event_type == EventType.VOTE and target is not None:
                self.bayesian_engine.update(RBAEventType.PLAYER_VOTED, actor_id=actor)
                self.bayesian_engine.update(RBAEventType.PLAYER_WAS_VOTED_FOR, actor_id=target)

            # Handle claims and accusations
            if event.event_type == EventType.CLAIM and event.value is not None:
                if actor not in self.player_stats: continue
                self.player_stats[actor]["claimed_role"] = event.value
                
                # Accusation
                if target is not None and target != actor:
                    # Storing claim for later verification
                    self.public_claims.append({"claimerId": actor, "targetId": target, "day": event.day})
                    
                    # An accusation is evidence against the accuser (if they are wrong) and the target
                    self.bayesian_engine.update(RBAEventType.PLAYER_ACCUSED, actor_id=actor)
                    
                    # The accusation also increases suspicion on the target, but less directly.
                    # This is now handled by ACCUSATION_WAS_RIGHT/WRONG after the fact.

        # Sync our internal knowledge with the engine
        for p_id in self.known_mafia:
            self.bayesian_engine.set_player_probability(p_id, is_mafia=True)
        for p_id in self.known_safe:
            self.bayesian_engine.set_player_probability(p_id, is_mafia=False)

    def _act_discussion(self, status: GameStatus) -> GameAction:
        survivors = [p for p in status.players if p.alive and p.id != self.id]
        if not survivors:
            return GameAction(target_id=-1)

        probabilities = self.bayesian_engine.get_mafia_probabilities()
        
        # Find the player with the highest suspicion, excluding self and known allies
        personal_target = -1
        max_personal_sus = -1
        for p in survivors:
            # Mafia shouldn't target other mafias
            if self.role == Role.MAFIA and p.id in self.known_mafia:
                continue
            
            sus = probabilities[p.id]
            if sus > max_personal_sus:
                max_personal_sus = sus
                personal_target = p.id

        # Estimate suspicion on myself from a townie's perspective.
        # This is a simplification; a true "theory of mind" model would be more complex.
        # We'll just use the raw probability for now.
        my_mafia_prob = probabilities[self.id]

        # 1. MAFIA
        if self.role == Role.MAFIA:
            # If suspicion on me is high, try to claim a role to deflect.
            if my_mafia_prob > 0.5 and not self.claimed_role:
                gamble = random.random()
                if gamble < 0.4: return self._do_claim(Role.POLICE)
                elif gamble < 0.7: return self._do_claim(Role.DOCTOR)
                else: return self._do_claim(Role.CITIZEN)
            # If there's a high-suspicion target, accuse them.
            elif personal_target != -1 and max_personal_sus > 0.6:
                return GameAction(target_id=personal_target, claim_role=Role.MAFIA)

        # 2. POLICE
        elif self.role == Role.POLICE:
            # If there's a fake police, call them out.
            counter = [p for p in survivors if self.player_stats.get(p.id, {}).get("claimed_role") == Role.POLICE]
            if counter:
                self.is_hiding_role = False
                return GameAction(target_id=counter[0].id, claim_role=Role.MAFIA)
            
            # If we found a mafia, accuse them.
            found_mafia = [m for m in self.known_mafia if m in [p.id for p in survivors]]
            if found_mafia:
                self.is_hiding_role = False
                return GameAction(target_id=found_mafia[0], claim_role=Role.MAFIA)
            
            # If not hiding, claim our role.
            if not self.is_hiding_role and not self.claimed_role:
                return self._do_claim(Role.POLICE)

        # 3. DOCTOR
        elif self.role == Role.DOCTOR:
            # If we made a successful save, we can reveal ourselves.
            if self.last_night_healed_id != -1 and self.player_stats.get(self.last_night_healed_id, {}).get("is_confirmed"):
                self.is_hiding_role = False
                return self._do_claim(Role.DOCTOR)
            
            # If someone is falsely claiming to be a doctor, counter-claim.
            counter = [p for p in survivors if self.player_stats.get(p.id, {}).get("claimed_role") == Role.DOCTOR]
            if counter and not self.claimed_role:
                return self._do_claim(Role.DOCTOR)

            if not self.is_hiding_role and not self.claimed_role:
                 return self._do_claim(Role.DOCTOR)

        # 4. CITIZEN
        elif self.role == Role.CITIZEN:
             if personal_target != -1 and max_personal_sus > 0.7:
                 if random.random() < 0.5:
                     return GameAction(target_id=personal_target, claim_role=Role.MAFIA)
        
        return GameAction(target_id=-1)

    def _act_vote(self, status: GameStatus) -> GameAction:
        alive_players = [p.id for p in status.players if p.alive]
        if not alive_players:
            return GameAction(target_id=-1)

        # Get mafia probabilities from the Bayesian engine
        probabilities = self.bayesian_engine.get_mafia_probabilities()

        # Mask probabilities of self, known safe players, and known mafia (if agent is mafia)
        mask = np.ones_like(probabilities)
        
        # Cannot vote for self
        mask[self.id] = 0
        
        # Do not vote for known safe players
        for safe_id in self.known_safe:
            if safe_id in alive_players:
                mask[safe_id] = 0
        
        # Mafia should not vote for other mafia unless they are highly suspicious to others (advanced strategy not implemented here)
        if self.role == Role.MAFIA:
            for mafia_id in self.known_mafia:
                if mafia_id in alive_players:
                    mask[mafia_id] = 0
        
        # Also, do not vote for dead players
        for i in range(len(probabilities)):
            if i not in alive_players:
                mask[i] = 0

        masked_probabilities = probabilities * mask
        
        # If all options are masked, abstain
        if np.sum(masked_probabilities) == 0:
            return GameAction(target_id=-1)
            
        # Normalize masked probabilities to use for entropy and softmax
        normalized_probs = masked_probabilities / np.sum(masked_probabilities)

        # Decide whether to abstain based on uncertainty (entropy)
        entropy = calculate_entropy(normalized_probs)
        if should_abstain(entropy, ABSTAIN_ENTROPY_THRESHOLD):
            return GameAction(target_id=-1)

        # Select a target using softmax selection
        target_id = softmax_selection(normalized_probs, temperature=VOTE_TEMPERATURE)
        
        return GameAction(target_id=target_id)

    def _act_execute(self, status: GameStatus) -> GameAction:
        # Find the player who was voted to be executed
        votes = {}
        for e in status.action_history:
            if e.day == status.day and e.phase == Phase.DAY_VOTE and e.event_type == EventType.VOTE:
                 if e.target_id is not None and e.target_id != -1:
                    votes[e.target_id] = votes.get(e.target_id, 0) + 1
        
        if not votes:
            return GameAction(target_id=-1)
        
        target_id = max(votes, key=votes.get)

        # Never agree to execute yourself
        if target_id == self.id:
            return GameAction(target_id=-1)

        # Mafia do not execute other mafia unless it's a strategic sacrifice (not implemented)
        if self.role == Role.MAFIA and target_id in self.known_mafia:
            return GameAction(target_id=-1)
            
        # Do not execute known safe players
        if target_id in self.known_safe:
            return GameAction(target_id=-1)

        # Get the probability of the target being mafia
        probabilities = self.bayesian_engine.get_mafia_probabilities()
        mafia_prob = probabilities[target_id]

        # Agree to execute if the probability is high enough
        if mafia_prob > 0.5: # 50% threshold
            return GameAction(target_id=target_id)
        
        return GameAction(target_id=-1)

    def _act_night(self, status: GameStatus, targets: List[int]) -> GameAction:
        alive_player_ids = [p.id for p in status.players if p.alive]

        if self.role == Role.MAFIA:
            # Mafia targets non-mafia players.
            if not self.known_mafia: self.known_mafia = []
            
            # Avoid hitting healers who have revealed themselves by claiming Doctor and self-healing.
            healers = [pid for pid, st in self.player_stats.items() 
                       if st.get("claimed_role") == Role.DOCTOR and st.get("declared_self_heal")]
            avoid = set(healers + self.known_mafia)
            
            # Prioritize revealed police or proven citizens
            revealed_police = [p_id for p_id in alive_player_ids if self.player_stats.get(p_id, {}).get("claimed_role") == Role.POLICE and p_id not in avoid]
            if revealed_police: return GameAction(target_id=revealed_police[0])
            
            proven_citizens = [p_id for p_id in self.known_safe if p_id in alive_player_ids and p_id not in avoid]
            if proven_citizens: return GameAction(target_id=proven_citizens[0])

            # Otherwise, choose a random player who is not a known ally.
            candidates = [p_id for p_id in alive_player_ids if p_id not in avoid]
            if candidates: return GameAction(target_id=random.choice(candidates))
            
        elif self.role == Role.DOCTOR:
            # Doctor prioritizes healing confirmed important roles, then themselves.
            claimed_police = [p_id for p_id in alive_player_ids if self.player_stats.get(p_id, {}).get("claimed_role") == Role.POLICE]
            if claimed_police: return GameAction(target_id=claimed_police[0])
            
            if self.declared_self_heal or random.random() < 0.2: return GameAction(target_id=self.id)
            
            proven = [p_id for p_id in self.known_safe if p_id in alive_player_ids]
            if proven: return GameAction(target_id=proven[0])
            
            return GameAction(target_id=self.id)

        elif self.role == Role.POLICE:
            # Police investigates the most suspicious player not yet investigated.
            probabilities = self.bayesian_engine.get_mafia_probabilities()
            
            unknown = [p_id for p_id in alive_player_ids if p_id not in self.investigated_log and p_id != self.id]
            if unknown:
                # Sort by mafia probability, highest first
                unknown.sort(key=lambda x: probabilities[x], reverse=True)
                target = unknown[0]
                self.investigated_log.append(target)
                return GameAction(target_id=target)
        
        return GameAction(target_id=-1)

    def _do_claim(self, role: Role) -> GameAction:
        self.claimed_role = role
        return GameAction(target_id=self.id, claim_role=role)