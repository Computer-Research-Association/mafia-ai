import random
from typing import List, Dict, Optional, Set
from core.agents.base_agent import BaseAgent
from core.engine.state import GameStatus, GameAction
from config import EventType, Role, Phase

class RuleBaseAgent(BaseAgent):
    def __init__(self, player_id: int, role: Role):
        super().__init__(player_id, role)
        self.fake_role = None 
        self.last_day = 0

    def get_action(self, status: GameStatus) -> GameAction:
        # [새 게임 감지]
        if status.day < self.last_day:
            self.fake_role = None
        self.last_day = status.day
        
        self.role = status.my_role
        self.id = status.my_id
        
        # 마피아 위장 직업 설정
        if self.role == Role.MAFIA and self.fake_role is None:
             self.fake_role = random.choice([Role.POLICE, Role.DOCTOR])

        if not any(p.id == self.id and p.alive for p in status.players):
            return GameAction(target_id=-1)

        alive_others = [p.id for p in status.players if p.alive and p.id != self.id]
        if not alive_others: return GameAction(target_id=-1)

        # 현재 상황 분석
        context = self._analyze_history(status)

        if status.phase == Phase.DAY_DISCUSSION: 
            return self._act_discussion(status, context)
        if status.phase == Phase.DAY_VOTE:       
            return self._act_vote(status, context, alive_others)
        if status.phase == Phase.DAY_EXECUTE:    
            return self._act_execute(status)
        if status.phase == Phase.NIGHT:          
            return self._act_night(status, alive_others, context)
        
        return GameAction(target_id=-1)

    def _analyze_history(self, status: GameStatus):
        """게임 로그 분석"""
        claims = {}
        accusations = {}
        investigations = {}
        known_roles = {}
        teammates = []
        my_claim_done_today = False
        my_accusation_done_today = False

        for event in status.action_history:
            if event.day == status.day and event.phase == Phase.DAY_DISCUSSION and event.actor_id == self.id:
                if event.event_type == EventType.CLAIM:
                    if event.target_id == self.id or event.target_id == -1:
                        my_claim_done_today = True
                    elif event.value == Role.MAFIA:
                        my_accusation_done_today = True

            if event.event_type == EventType.CLAIM:
                if event.target_id == event.actor_id or event.target_id == -1:
                    if event.value: claims[event.actor_id] = event.value
                elif event.target_id != event.actor_id and event.value == Role.MAFIA:
                    accusations[event.actor_id] = event.target_id
            
            if event.event_type == EventType.EXECUTE:
                known_roles[event.target_id] = event.value 
            
            if event.event_type == EventType.POLICE_RESULT and event.actor_id == self.id:
                investigations[event.target_id] = event.value
                if self.role == Role.POLICE:
                    known_roles[event.target_id] = event.value

            if self.role == Role.MAFIA:
                if event.event_type == EventType.POLICE_RESULT and event.actor_id == -1:
                    if event.value == Role.MAFIA and event.target_id != self.id:
                        if event.target_id not in teammates:
                            teammates.append(event.target_id)
        
        seed = getattr(status, 'random_seed', 0)
        
        return {
            "claims": claims,
            "accusations": accusations,
            "investigations": investigations,
            "known_roles": known_roles,
            "teammates": teammates,
            "my_claim_done_today": my_claim_done_today,
            "my_accusation_done_today": my_accusation_done_today,
            "seed": seed
        }

    def _pick_fair_deterministic(self, candidates: List[int], day: int, seed: int) -> int:
        """
        [공정한 결정론적 선택]
        1. 후보 리스트를 정렬 (기본 순서 보장)
        2. 공유된 시드(seed)를 사용해 리스트를 셔플 (순서 랜덤화)
        3. 날짜(day)에 따라 순환 선택 (Day 1에 누가 죽을지 매판 달라짐)
        """
        if not candidates: return -1
        
        # 1. 정렬
        candidates.sort()
        
        # 2. 공유 시드로 셔플 (모든 에이전트가 동일한 결과를 얻음)
        # (local random instance를 사용하여 전역 random 상태에 영향 안 줌)
        rng = random.Random(seed)
        rng.shuffle(candidates)
        
        # 3. 날짜 기반 선택
        return candidates[day % len(candidates)]

    def _get_confirmed_players(self, alive_players: List[int], context: dict) -> Set[int]:
        confirmed = set()
        claims = context["claims"]
        known_roles = context["known_roles"]

        for role_type in [Role.POLICE, Role.DOCTOR]:
            alive_claimants = [pid for pid, r in claims.items() if r == role_type and pid in alive_players]
            dead_real = [pid for pid, r in known_roles.items() if r == role_type]

            if len(alive_claimants) == 1 and not dead_real:
                confirmed.add(alive_claimants[0])
        return confirmed

    def _get_logic_target(self, alive_players: List[int], context: dict, day: int) -> Optional[int]:
        seed = context["seed"]

        # 0. [자살 방지] 마피아 커밍아웃 시 0순위 척살
        mafia_candidates = []
        for pid, role in context["claims"].items():
            if role == Role.MAFIA and pid in alive_players:
                if pid != self.id: mafia_candidates.append(pid)
        if mafia_candidates:
            return self._pick_fair_deterministic(mafia_candidates, day, seed)

        # 1. [절대 정보] 내 조사 결과 마피아
        if self.role == Role.POLICE:
            found_mafias = []
            for target, result in context["investigations"].items():
                if result == Role.MAFIA and target in alive_players:
                    found_mafias.append(target)
            if found_mafias: 
                return self._pick_fair_deterministic(found_mafias, day, seed)
        
        # 2. [소거법] 팩트 기반 추론
        confirmed_fake = []
        for role_type in [Role.POLICE, Role.DOCTOR]:
            claimants = [pid for pid, r in context["claims"].items() if r == role_type]
            true_role_dead = [pid for pid in claimants if context["known_roles"].get(pid) == role_type]
            if true_role_dead:
                for suspect in claimants:
                    if suspect not in true_role_dead and suspect in alive_players:
                        if suspect != self.id: confirmed_fake.append(suspect)
        if confirmed_fake: 
            return self._pick_fair_deterministic(confirmed_fake, day, seed)

        # 3. [확경 신뢰]
        confirmed_players = self._get_confirmed_players(alive_players, context)
        police_claims = [pid for pid, role in context["claims"].items() if role == Role.POLICE and pid in alive_players]
        
        if len(police_claims) == 1:
            cop_id = police_claims[0]
            if cop_id in confirmed_players and cop_id in context["accusations"]:
                target = context["accusations"][cop_id]
                if target in alive_players and target != self.id:
                    return target

        # 4. [대립 정리] 맞경 -> 맞의 (공정 셔플 후 선택)
        police_rivals = [p for p in police_claims if p != self.id]
        if len(police_claims) >= 2 and police_rivals:
            return self._pick_fair_deterministic(police_rivals, day, seed)
        
        doctor_claims = [pid for pid, role in context["claims"].items() if role == Role.DOCTOR and pid in alive_players]
        doctor_rivals = [p for p in doctor_claims if p != self.id]
        if len(doctor_claims) >= 2 and doctor_rivals:
            return self._pick_fair_deterministic(doctor_rivals, day, seed)

        return None

    def _act_discussion(self, status: GameStatus, context: dict) -> GameAction:
        alive_players = [p.id for p in status.players if p.alive]

        # 1. Day 1 직업 공개
        if status.day == 1 and not context["my_claim_done_today"]:
            if self.role != Role.CITIZEN:
                claim = self.fake_role if self.role == Role.MAFIA else self.role
                return GameAction(target_id=self.id, claim_role=claim)

        # 2. 확정 특수직 있으면 시민 침묵
        if self.role == Role.CITIZEN:
            confirmed = self._get_confirmed_players(alive_players, context)
            if confirmed: return GameAction(target_id=-1)

        # 3. 논리적 타겟 지목
        target = self._get_logic_target(alive_players, context, status.day)
        if target is not None and not context["my_accusation_done_today"]:
            return GameAction(target_id=target, claim_role=Role.MAFIA)
        
        return GameAction(target_id=-1)

    def _act_vote(self, status: GameStatus, context: dict, alive_others: List[int]) -> GameAction:
        alive_players = [p.id for p in status.players if p.alive]
        seed = context["seed"]

        target = self._get_logic_target(alive_players, context, status.day)
        if target is not None:
            return GameAction(target_id=target)

        # 엔드게임: 랜덤(공정 셔플) 투표
        if len(alive_players) <= 4:
            confirmed = self._get_confirmed_players(alive_players, context)
            candidates = [p for p in alive_others if p not in confirmed]
            
            if candidates:
                return GameAction(target_id=self._pick_fair_deterministic(candidates, status.day, seed))
            elif alive_others:
                 return GameAction(target_id=self._pick_fair_deterministic(alive_others, status.day, seed))

        return GameAction(target_id=-1)

    def _act_execute(self, status: GameStatus) -> GameAction:
        votes = {}
        for e in status.action_history:
            if e.day == status.day and e.phase == Phase.DAY_VOTE and e.event_type == EventType.VOTE:
                if e.target_id not in [None, -1]:
                    votes[e.target_id] = votes.get(e.target_id, 0) + 1
        
        if not votes: return GameAction(target_id=self.id)
        
        target_id = max(votes, key=votes.get)
        if target_id == self.id or target_id == -1: return GameAction(target_id=self.id)
        
        return GameAction(target_id=target_id)

    def _act_night(self, status: GameStatus, alive_others: List[int], context: dict) -> GameAction:
        seed = context["seed"]
        if self.role == Role.MAFIA:
            special_roles = [pid for pid, role in context["claims"].items() if role in [Role.POLICE, Role.DOCTOR]]
            exclude = special_roles + context["teammates"]
            silent_citizens = [t for t in alive_others if t not in exclude]
            
            if silent_citizens: 
                return GameAction(target_id=self._pick_fair_deterministic(silent_citizens, status.day, seed))
            
            valid = [t for t in alive_others if t not in context["teammates"]]
            if valid: 
                return GameAction(target_id=self._pick_fair_deterministic(valid, status.day, seed))
            return GameAction(target_id=-1)

        if self.role == Role.POLICE:
            rivals = [pid for pid, role in context["claims"].items() if role == Role.POLICE and pid != self.id and pid in alive_others]
            if rivals: return GameAction(target_id=self._pick_fair_deterministic(rivals, status.day, seed))
            
            doctor_claims = [pid for pid, role in context["claims"].items() if role == Role.DOCTOR and pid in alive_others]
            if len(doctor_claims) >= 2:
                uninvestigated_docs = [t for t in doctor_claims if t not in context["investigations"]]
                if uninvestigated_docs:
                    return GameAction(target_id=self._pick_fair_deterministic(uninvestigated_docs, status.day, seed))

            confirmed = self._get_confirmed_players(alive_others, context)
            uninvestigated = [t for t in alive_others if t not in context["investigations"] and t not in confirmed]
            
            if uninvestigated: return GameAction(target_id=self._pick_fair_deterministic(uninvestigated, status.day, seed))
            return GameAction(target_id=self._pick_fair_deterministic(alive_others, status.day, seed))

        if self.role == Role.DOCTOR:
            dead_real_police = [pid for pid, r in context["known_roles"].items() if r == Role.POLICE]
            if dead_real_police:
                return GameAction(target_id=self.id)

            police_claims = [pid for pid, role in context["claims"].items() if role == Role.POLICE and pid in alive_others]
            if police_claims: return GameAction(target_id=self._pick_fair_deterministic(police_claims, status.day, seed))
            
            return GameAction(target_id=self.id)

        return GameAction(target_id=-1)