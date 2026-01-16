import random
from typing import List, Dict, Optional
from core.agents.base_agent import BaseAgent
from core.engine.state import GameStatus, GameAction
from config import EventType, Role, Phase

# [Tuning] 시민 단합력을 위한 상수 조정
BASE_SUSPICION = 10.0
BASE_ATTENTION = 10.0
VOTE_THRESHOLD_BASE = 5.0

class RuleBaseAgent(BaseAgent):
    def __init__(self, player_id: int, role: Role):
        super().__init__(player_id, role)
        self.reset()

    def reset(self):
        self.suspicion = BASE_SUSPICION
        self.attention = BASE_ATTENTION
        self.individual_suspicion: Dict[int, float] = {} 
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
        # 1. Init
        if not self.player_stats:
            for p in status.players:
                self.player_stats[p.id] = {
                    "suspicion": BASE_SUSPICION,
                    "attention": BASE_ATTENTION,
                    "claimed_role": None,
                    "is_confirmed": False,
                    "is_proven_citizen": False,
                    "declared_self_heal": False
                }
                if p.id not in self.individual_suspicion:
                    self.individual_suspicion[p.id] = 10.0

        # 2. Reset Global Stats
        for pid in self.player_stats:
            self.player_stats[pid]["suspicion"] = BASE_SUSPICION
            self.player_stats[pid]["attention"] = BASE_ATTENTION
        
        for pid in self.individual_suspicion:
            self.individual_suspicion[pid] = 10.0
            if self.role == Role.MAFIA and pid in self.known_mafia:
                 self.individual_suspicion[pid] = 0.0

        self.public_claims = []
        
        dead_roles = {Role.POLICE: 0, Role.DOCTOR: 0}
        alive_claims = {Role.POLICE: [], Role.DOCTOR: []}

        # 3. Replay
        for event in status.action_history:
            actor = event.actor_id
            target = event.target_id # [주의] 여기서 target이 None일 수 있음

            # [FIX] target이 None이면 -1로 처리하거나 건너뛰기
            # 유효한 타겟(플레이어 ID)이 아니면 통계 갱신 로직을 스킵해야 안전함
            if target is None: 
                target = -1
            
            # --- Dead Role Counting ---
            if event.event_type == EventType.EXECUTE and event.value:
                if event.value in dead_roles: dead_roles[event.value] += 1
            
            # --- Claims ---
            if event.event_type == EventType.CLAIM and event.value is not None:
                # actor나 target이 유효하지 않으면 스킵
                if actor not in self.player_stats or (target != -1 and target not in self.player_stats):
                    continue

                if actor == target: # Self Claim
                    claimed_role = event.value
                    self.player_stats[actor]["claimed_role"] = claimed_role
                    
                    if claimed_role in alive_claims:
                        if actor not in alive_claims[claimed_role]: alive_claims[claimed_role].append(actor)

                    is_fake = False
                    if claimed_role in dead_roles and dead_roles[claimed_role] >= 1: is_fake = True
                    
                    if self.role == claimed_role and claimed_role != Role.CITIZEN and actor != self.id:
                        self.individual_suspicion[actor] = 999.0 
                    
                    if len(alive_claims.get(claimed_role, [])) >= 2 and claimed_role != Role.CITIZEN:
                        for claimer in alive_claims[claimed_role]:
                            self.player_stats[claimer]["suspicion"] += 100.0 
                            self.player_stats[claimer]["attention"] += 50.0

                    if is_fake:
                        self.player_stats[actor]["suspicion"] += 999.0
                        self.individual_suspicion[actor] = 999.0

                    if claimed_role != Role.CITIZEN:
                        rivals = [pid for pid, st in self.player_stats.items() 
                                  if pid != actor and st["claimed_role"] == claimed_role]
                        if rivals:
                            self.player_stats[actor]["suspicion"] += 50
                            for r in rivals: self.player_stats[r]["suspicion"] += 50

                else: # Accusation
                    # 타겟이 유효한 플레이어일 때만 처리
                    if target != -1:
                        res_str = 'MAFIA' if event.value == Role.MAFIA else 'CITIZEN'
                        self.public_claims.append({
                            "claimerId": actor, "targetId": target, 
                            "role": self.player_stats[actor]["claimed_role"], 
                            "result": res_str, "day": event.day
                        })
                        
                        trust_score = 100 - self.individual_suspicion.get(actor, 50)
                        if self.player_stats[actor]["is_confirmed"]: trust_score = 100
                        
                        if trust_score > 80: 
                            if event.value == Role.MAFIA:
                                self.individual_suspicion[target] += 500.0 
                            elif event.value == Role.CITIZEN:
                                self.individual_suspicion[target] = 0.0

                        if event.value == Role.MAFIA:
                            self.player_stats[target]["suspicion"] += 40
                            self.player_stats[target]["attention"] += 20
                        elif event.value == Role.CITIZEN:
                            reduce = 1000 if self.player_stats[actor]["is_confirmed"] else 30
                            # [FIX] 여기서 KeyError 발생했던 지점
                            self.player_stats[target]["suspicion"] = max(0, self.player_stats[target]["suspicion"] - reduce)
                            if self.player_stats[actor]["is_confirmed"]:
                                self.player_stats[target]["is_proven_citizen"] = True
            
            # --- Execution & Feedback ---
            if event.event_type == EventType.EXECUTE:
                 # target이 -1이거나 None이면 스킵
                 if event.value is None or target == -1: continue
                 
                 for r in alive_claims:
                     if target in alive_claims[r]: alive_claims[r].remove(target)

                 if event.value != Role.MAFIA: 
                     for c in self.public_claims:
                         if c["targetId"] == target and c["result"] == 'MAFIA':
                             self.player_stats[c["claimerId"]]["suspicion"] += 80 
                             self.individual_suspicion[c["claimerId"]] = self.individual_suspicion.get(c["claimerId"], 10) + 80
                 else: 
                     for c in self.public_claims:
                         if c["targetId"] == target and c["result"] == 'MAFIA':
                             c_role = self.player_stats[c["claimerId"]]["claimed_role"]
                             if c_role == Role.POLICE:
                                 self.player_stats[c["claimerId"]]["is_confirmed"] = True
                                 self.player_stats[c["claimerId"]]["suspicion"] = 0
                                 self.individual_suspicion[c["claimerId"]] = 0.0

            # --- Internal Knowledge ---
            if event.event_type == EventType.POLICE_RESULT:
                reliable = (self.role==Role.POLICE and actor==self.id) or (self.role==Role.MAFIA and actor==-1)
                # target 유효성 체크
                if reliable and target != -1:
                    is_mafia = (event.value == Role.MAFIA)
                    if is_mafia:
                        target_list = self.known_mafia if self.role==Role.MAFIA else self.known_mafia
                        if target not in target_list: target_list.append(target)
                        if self.role != Role.MAFIA: self.individual_suspicion[target] = 999.0
                    else:
                        if target not in self.known_safe: self.known_safe.append(target)
                        self.individual_suspicion[target] = 0.0

    def _act_discussion(self, status: GameStatus) -> GameAction:
        survivors = [p for p in status.players if p.alive and p.id != self.id]
        personal_target = None
        max_personal_sus = -1
        for p in survivors:
            sus = self.individual_suspicion.get(p.id, 10)
            if sus > max_personal_sus:
                max_personal_sus = sus
                personal_target = p

        my_sus = self.player_stats[self.id]["suspicion"]
        
        # 1. MAFIA
        if self.role == Role.MAFIA:
            if my_sus > 70 and not self.claimed_role:
                gamble = random.random()
                if gamble < 0.4: return self._do_claim(Role.POLICE)
                elif gamble < 0.7: return self._do_claim(Role.DOCTOR)
                else: return self._do_claim(Role.CITIZEN)
            elif not self.claimed_role and random.random() < 0.15: 
                if status.day == 1: return self._do_claim(Role.POLICE)
                elif random.random() < 0.6:
                    targets = [p for p in survivors if p.id not in self.known_mafia]
                    if targets: return GameAction(target_id=random.choice(targets).id, claim_role=Role.MAFIA)

        # 2. POLICE
        elif self.role == Role.POLICE:
            counter = [p for p in survivors if self.player_stats[p.id]["claimed_role"] == Role.POLICE]
            if counter:
                self.is_hiding_role = False
                return GameAction(target_id=counter[0].id, claim_role=Role.MAFIA)
            
            found_mafia = [m for m in self.known_mafia if m in [p.id for p in survivors]]
            if found_mafia:
                self.is_hiding_role = False
                return GameAction(target_id=found_mafia[0], claim_role=Role.MAFIA)
            
            if not self.is_hiding_role and not self.claimed_role:
                return self._do_claim(Role.POLICE)

        # 3. DOCTOR
        elif self.role == Role.DOCTOR:
            if self.last_night_healed_id != -1:
                self.is_hiding_role = False
                return self._do_claim(Role.DOCTOR)
            
            counter = [p for p in survivors if self.player_stats[p.id]["claimed_role"] == Role.DOCTOR]
            if counter and not self.claimed_role:
                return self._do_claim(Role.DOCTOR)

            if not self.is_hiding_role and not self.claimed_role:
                 return self._do_claim(Role.DOCTOR)

        # 4. CITIZEN
        elif self.role == Role.CITIZEN:
             if personal_target and max_personal_sus > 100:
                 if random.random() < 0.5:
                     return GameAction(target_id=personal_target.id, claim_role=Role.MAFIA)
        
        return GameAction(target_id=-1)

    def _act_vote(self, status: GameStatus) -> GameAction:
        survivors = [p for p in status.players if p.alive]
        death_rate = (len(status.players) - len(survivors)) / len(status.players)
        vote_threshold = max(3, VOTE_THRESHOLD_BASE - (death_rate * 30))
        
        best_target = -1
        max_score = -9999.0

        for candidate in survivors:
            if candidate.id == self.id: continue
            
            cand_stat = self.player_stats[candidate.id]
            ind_sus = self.individual_suspicion.get(candidate.id, 10)
            
            score = ind_sus * 5.0 + cand_stat["suspicion"] * 1.0 + cand_stat["attention"] * 0.5
            score += random.random() * 0.1
            
            is_safe = False
            if self.role == Role.MAFIA and candidate.id in self.known_mafia: is_safe = True
            if candidate.id in self.known_safe: is_safe = True
            
            if is_safe:
                if self.role == Role.MAFIA and cand_stat["suspicion"] > 150: score += 50
                else: score = -9999.0

            if score > max_score:
                max_score = score
                best_target = candidate.id
        
        if max_score > vote_threshold:
            return GameAction(target_id=best_target)
        return GameAction(target_id=-1)

    def _act_execute(self, status: GameStatus) -> GameAction:
        target_id = -1
        votes = {}
        for e in status.action_history:
            if e.day == status.day and e.phase == Phase.DAY_VOTE and e.event_type == EventType.VOTE:
                 if e.target_id != -1: votes[e.target_id] = votes.get(e.target_id, 0) + 1
        
        if not votes: return GameAction(target_id=-1)
        target_id = max(votes, key=votes.get)
        
        cand_stat = self.player_stats[target_id]
        survivors = [p for p in status.players if p.alive]
        
        my_sus_on_target = self.individual_suspicion.get(target_id, 10)
        
        if my_sus_on_target > 90: return GameAction(target_id=target_id)
        
        if my_sus_on_target < 5: return GameAction(target_id=-1)
        if target_id in self.known_safe: return GameAction(target_id=-1)
        if cand_stat["is_proven_citizen"]: return GameAction(target_id=-1)
        
        chance = 0.5
        if cand_stat["suspicion"] > 50: chance += 0.4
        
        target_role_claim = cand_stat["claimed_role"]
        if target_role_claim and target_role_claim != Role.CITIZEN:
             if any(self.player_stats[p.id]["claimed_role"] == target_role_claim for p in survivors if p.id != target_id):
                 chance += 0.5

        agree = (chance > 0.5)
        
        if self.role == Role.MAFIA:
            if target_id in self.known_mafia: agree = (cand_stat["suspicion"] > 150)
            else: agree = True
        
        if target_id == self.id: agree = False
        
        return GameAction(target_id=target_id) if agree else GameAction(target_id=-1)

    def _act_night(self, status: GameStatus, targets: List[int]) -> GameAction:
        if self.role == Role.MAFIA:
            if not self.known_mafia: self.known_mafia = []
            healers = [pid for pid, st in self.player_stats.items() 
                       if st["claimed_role"] == Role.DOCTOR and st["declared_self_heal"]]
            avoid = set(healers + self.known_mafia)
            
            revealed_police = [p for p in targets if self.player_stats[p]["claimed_role"] == Role.POLICE and p not in avoid]
            if revealed_police: return GameAction(target_id=revealed_police[0])
            
            proven = [p for p in targets if self.player_stats[p]["is_proven_citizen"] and p not in avoid]
            if proven: return GameAction(target_id=proven[0])

            candidates = [p for p in targets if p not in avoid]
            if candidates: return GameAction(target_id=random.choice(candidates))
            
        elif self.role == Role.DOCTOR:
            claimed_police = [p for p in targets if self.player_stats[p]["claimed_role"] == Role.POLICE]
            if claimed_police: return GameAction(target_id=claimed_police[0])
            
            if self.declared_self_heal or random.random() < 0.2: return GameAction(target_id=self.id)
            
            proven = [p for p in targets if self.player_stats[p]["is_proven_citizen"]]
            if proven: return GameAction(target_id=proven[0])
            
            return GameAction(target_id=self.id)

        elif self.role == Role.POLICE:
            unknown = [t for t in targets if t not in self.investigated_log and t != self.id]
            if unknown:
                unknown.sort(key=lambda x: self.player_stats[x]["suspicion"], reverse=True)
                target = unknown[0]
                self.investigated_log.append(target)
                return GameAction(target_id=target)
        
        return GameAction(target_id=-1)

    def _do_claim(self, role: Role) -> GameAction:
        self.claimed_role = role
        return GameAction(target_id=self.id, claim_role=role)