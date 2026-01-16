import random
from typing import List, Set, Dict
from core.agents.base_agent import BaseAgent
from core.engine.state import GameStatus, GameAction
from config import EventType, Role, Phase

class RuleBaseAgent(BaseAgent):
    def __init__(self, player_id: int, role: Role):
        super().__init__(player_id, role)
        self.suspicion_scores: Dict[int, float] = {}
        self.attention_scores: Dict[int, float] = {}
        self.role_claims: Dict[int, Role] = {}
        self.proven_citizens: Set[int] = {self.id}
        self.known_mafia: Set[int] = set()
        self.mafia_teammates: Set[int] = set()
        self.investigated_players: Set[int] = set()
        self.has_claimed_role: bool = False

    def reset(self):
        self.suspicion_scores.clear()
        self.attention_scores.clear()
        self.role_claims.clear()
        self.proven_citizens = {self.id}
        self.known_mafia.clear()
        self.mafia_teammates.clear()
        self.investigated_players.clear()
        self.has_claimed_role = False

    def get_action(self, status: GameStatus) -> GameAction:
        self.role = status.my_role
        self._update_knowledge(status)
        alive_others = [p.id for p in status.players if p.alive and p.id != self.id]
        if not alive_others: return GameAction(target_id=-1)

        if status.phase == Phase.DAY_DISCUSSION:
            return self._act_discussion(status, alive_others)
        elif status.phase == Phase.DAY_VOTE:
            return self._act_vote(status, alive_others)
        elif status.phase == Phase.DAY_EXECUTE:
            return self._act_execute(status, alive_others)
        elif status.phase == Phase.NIGHT:
            return self._act_night(status, alive_others)
        return GameAction(target_id=-1)

    def _update_knowledge(self, status: GameStatus):
        self.suspicion_scores = {p.id: 10.0 for p in status.players}
        self.attention_scores = {p.id: 10.0 for p in status.players}
        self.role_claims = {}
        self.proven_citizens = {self.id}
        self.known_mafia = set()

        for event in status.action_history:
            if event.event_type == EventType.CLAIM:
                actor, target, val = event.actor_id, event.target_id, event.value
                if val is None or val == Role.CITIZEN: continue
                
                self.attention_scores[actor] += 5.0
                
                if actor == target:
                    self.role_claims[actor] = val
                    rivals = [pid for pid, role in self.role_claims.items() if role == val and pid != actor]
                    if rivals:
                        for r_id in rivals: self.suspicion_scores[r_id] += 30.0
                        self.suspicion_scores[actor] += 30.0
                else:
                    if val == Role.MAFIA:
                        self.suspicion_scores[target] += 40.0
                    elif val in [Role.POLICE, Role.DOCTOR]:
                        self.proven_citizens.add(target)
                        self.suspicion_scores[target] -= 20.0

            if event.event_type == EventType.VOTE and event.target_id == self.id:
                self.suspicion_scores[event.actor_id] += 15.0

            if event.event_type == EventType.POLICE_RESULT:
                is_reliable = (self.role == Role.POLICE and event.actor_id == self.id) or \
                              (self.role == Role.MAFIA and event.actor_id == -1)
                if is_reliable:
                    if event.value == Role.MAFIA:
                        if self.role == Role.MAFIA:
                            self.mafia_teammates.add(event.target_id)
                            self.suspicion_scores[event.target_id] = 0.0
                        else:
                            self.known_mafia.add(event.target_id)
                            self.suspicion_scores[event.target_id] = 100.0
                    else:
                        self.proven_citizens.add(event.target_id)
                        self.suspicion_scores[event.target_id] = 0.0
            
            if event.event_type == EventType.EXECUTE:
                if event.value == Role.MAFIA:
                    for prev in status.action_history:
                        if prev.day == event.day and prev.event_type == EventType.VOTE and prev.target_id == event.target_id:
                            self.suspicion_scores[prev.actor_id] -= 25.0
                elif event.value is not None:
                    for prev in status.action_history:
                        if prev.day == event.day and prev.event_type == EventType.VOTE and prev.target_id == event.target_id:
                            self.suspicion_scores[prev.actor_id] += 20.0
                        if prev.event_type == EventType.CLAIM and prev.target_id == event.target_id and prev.value == Role.MAFIA:
                            self.suspicion_scores[prev.actor_id] += 50.0

    def _act_discussion(self, status: GameStatus, targets: List[int]) -> GameAction:
        if self.role == Role.POLICE:
            if not self.has_claimed_role:
                if any(m in targets for m in self.known_mafia):
                    self.has_claimed_role = True
                    return GameAction(target_id=self.id, claim_role=Role.POLICE)
            for m_id in self.known_mafia:
                if m_id in targets: return GameAction(target_id=m_id, claim_role=Role.MAFIA)
        
        if self.role == Role.MAFIA:
            if self.suspicion_scores.get(self.id, 0) > 60 and not self.has_claimed_role:
                self.has_claimed_role = True
                return GameAction(target_id=self.id, claim_role=Role.POLICE)
            
            if not self.has_claimed_role and status.day <= 2 and random.random() < 0.4:
                potential_victims = [t for t in targets if t not in self.mafia_teammates]
                if potential_victims:
                    target = random.choice(potential_victims)
                    self.has_claimed_role = True
                    return GameAction(target_id=target, claim_role=Role.POLICE)

            if self.has_claimed_role:
                potential = [t for t in targets if t not in self.mafia_teammates]
                if potential: return GameAction(target_id=random.choice(potential), claim_role=Role.MAFIA)
        return GameAction(target_id=-1)

    def _act_vote(self, status: GameStatus, targets: List[int]) -> GameAction:
        if not targets: return GameAction(target_id=-1)
        valid = [t for t in targets if t not in self.mafia_teammates] if self.role == Role.MAFIA else targets
        if not valid: return GameAction(target_id=-1)

        if self.role != Role.MAFIA:
            for m_id in self.known_mafia:
                if m_id in valid: return GameAction(target_id=m_id)

        best, max_s = -1, -9999.0
        for tid in valid:
            s = self.suspicion_scores.get(tid, 10.0) * 1.5 + self.attention_scores.get(tid, 10.0) * 0.5 + random.uniform(-10, 30)
            if tid in self.proven_citizens: s -= 50.0
            if s > max_s: max_s, best = s, tid
        return GameAction(target_id=best) if max_s > 0 else GameAction(target_id=-1)

    def _act_execute(self, status: GameStatus, targets: List[int]) -> GameAction:
        counts = {}
        for e in status.action_history:
            if e.day == status.day and e.phase == Phase.DAY_VOTE and e.event_type == EventType.VOTE:
                if e.target_id != -1: counts[e.target_id] = counts.get(e.target_id, 0) + 1
        cid = -1
        if counts:
            mv = max(counts.values())
            tops = [tid for tid, c in counts.items() if c == mv]
            if len(tops) == 1: cid = tops[0]

        if cid != -1:
            chance = 0.55 + random.uniform(-0.15, 0.15)
            if self.suspicion_scores.get(cid, 10.0) <= sum(self.suspicion_scores.values())/len(self.suspicion_scores): chance -= 0.1
            if cid in self.proven_citizens or cid in self.mafia_teammates: chance = 0.0
            if chance > 0.5: return GameAction(target_id=cid)
        return GameAction(target_id=-1)

    def _act_night(self, status: GameStatus, targets: List[int]) -> GameAction:
        if self.role == Role.CITIZEN: return GameAction(target_id=-1)
        if self.role == Role.MAFIA:
            pot = [t for t in targets if t not in self.mafia_teammates]
            if not pot: return GameAction(target_id=-1)
            target = max(pot, key=lambda x: self.attention_scores.get(x, 0) - self.suspicion_scores.get(x, 0) + (20 if x in self.role_claims else 0) + random.uniform(-5, 5))
            return GameAction(target_id=target)
        if self.role == Role.DOCTOR:
            for pid, r in self.role_claims.items():
                if r == Role.POLICE and pid in targets: return GameAction(target_id=pid)
            
            potentials = targets + [self.id]
            best_target = max(potentials, key=lambda p: (self.attention_scores.get(p, 0) - self.suspicion_scores.get(p, 0) - (20.0 if p == self.id else 0)))
            return GameAction(target_id=best_target)
        if self.role == Role.POLICE:
            un = [t for t in targets if t not in self.investigated_players]
            if not un: return GameAction(target_id=-1)
            target = max(un, key=lambda x: self.suspicion_scores.get(x, 10) + random.uniform(-5, 5))
            self.investigated_players.add(target)
            return GameAction(target_id=target)
        return GameAction(target_id=-1)