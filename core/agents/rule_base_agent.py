import random
from typing import List, Dict, Optional, Set
from core.agents.base_agent import BaseAgent
from core.engine.state import GameStatus, GameAction
from config import EventType, Role, Phase

class RuleBaseAgent(BaseAgent):
    def __init__(self, player_id: int, role: Role):
        super().__init__(player_id, role)
        self.fake_role = None # 마피아 위장용
        self.last_day = 0

    def get_action(self, status: GameStatus) -> GameAction:
        # [새 게임 감지]
        if status.day < self.last_day:
            self.fake_role = None
        self.last_day = status.day
        
        self.role = status.my_role
        self.id = status.my_id
        
        # 마피아 위장 직업 설정 (없으면 생성)
        if self.role == Role.MAFIA and self.fake_role is None:
             self.fake_role = random.choice([Role.POLICE, Role.DOCTOR])

        # 죽은 자는 침묵
        if not any(p.id == self.id and p.alive for p in status.players):
            return GameAction(target_id=-1)

        alive_others = [p.id for p in status.players if p.alive and p.id != self.id]
        if not alive_others: return GameAction(target_id=-1)

        # 현재 상황 분석 (Stateless)
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
            # [중복 발언 방지 체크]
            if event.day == status.day and event.phase == Phase.DAY_DISCUSSION and event.actor_id == self.id:
                if event.event_type == EventType.CLAIM:
                    if event.target_id == self.id or event.target_id == -1:
                        my_claim_done_today = True
                    elif event.value == Role.MAFIA:
                        my_accusation_done_today = True

            # 1. 직업 공개 및 고발 파싱
            if event.event_type == EventType.CLAIM:
                # 직업 공개
                if event.target_id == event.actor_id or event.target_id == -1:
                    if event.value: claims[event.actor_id] = event.value
                # 타인 고발
                elif event.target_id != event.actor_id and event.value == Role.MAFIA:
                    accusations[event.actor_id] = event.target_id
            
            # 2. 처형 결과
            if event.event_type == EventType.EXECUTE:
                known_roles[event.target_id] = event.value 
            
            # 3. (경찰) 내 조사 결과
            if event.event_type == EventType.POLICE_RESULT and event.actor_id == self.id:
                investigations[event.target_id] = event.value
                if self.role == Role.POLICE:
                    known_roles[event.target_id] = event.value

            # 4. (마피아) 동료 확인
            if self.role == Role.MAFIA:
                if event.event_type == EventType.POLICE_RESULT and event.actor_id == -1:
                    if event.value == Role.MAFIA and event.target_id != self.id:
                        if event.target_id not in teammates:
                            teammates.append(event.target_id)
        
        return {
            "claims": claims,
            "accusations": accusations,
            "investigations": investigations,
            "known_roles": known_roles,
            "teammates": teammates,
            "my_claim_done_today": my_claim_done_today,
            "my_accusation_done_today": my_accusation_done_today
        }

    def _get_confirmed_players(self, alive_players: List[int], context: dict) -> Set[int]:
        """[확정 로직] 현재 상황에서 확실하게 경찰/의사로 믿을 수 있는 사람 식별"""
        confirmed = set()
        claims = context["claims"]
        known_roles = context["known_roles"]

        # 경찰/의사 각각에 대해
        for role_type in [Role.POLICE, Role.DOCTOR]:
            # 살아있는 주장자
            alive_claimants = [pid for pid, r in claims.items() if r == role_type and pid in alive_players]
            # 죽은 '진짜' (있다면)
            dead_real = [pid for pid, r in known_roles.items() if r == role_type]

            # 조건: 주장자가 딱 1명이고, 진짜가 죽지 않았을 때 -> 그 사람은 확정
            if len(alive_claimants) == 1 and not dead_real:
                confirmed.add(alive_claimants[0])
        
        return confirmed

    def _get_logic_target(self, alive_players: List[int], context: dict) -> Optional[int]:
        """[시민 행동 우선순위] 투표 및 토론 타겟 선정"""
        
        # 0. [자살 방지] 마피아 커밍아웃 시 0순위 척살
        for pid, role in context["claims"].items():
            if role == Role.MAFIA and pid in alive_players:
                if pid != self.id: return pid

        # 1. [절대 정보] 내 조사 결과 마피아 (경찰 전용)
        if self.role == Role.POLICE:
            for target, result in context["investigations"].items():
                if result == Role.MAFIA and target in alive_players:
                    return target
        
        # 2. [소거법] 팩트 기반 추론 (가장 강력한 시민 논리)
        for role_type in [Role.POLICE, Role.DOCTOR]:
            claimants = [pid for pid, r in context["claims"].items() if r == role_type]
            true_role_dead = [pid for pid in claimants if context["known_roles"].get(pid) == role_type]
            if true_role_dead:
                for suspect in claimants:
                    if suspect not in true_role_dead and suspect in alive_players:
                        if suspect != self.id: return suspect

        # 3. [확경 신뢰] 확정 경찰의 오더
        confirmed_players = self._get_confirmed_players(alive_players, context)
        police_claims = [pid for pid, role in context["claims"].items() if role == Role.POLICE and pid in alive_players]
        
        if len(police_claims) == 1:
            cop_id = police_claims[0]
            # 확경이 누군가를 지목했다면
            if cop_id in confirmed_players and cop_id in context["accusations"]:
                target = context["accusations"][cop_id]
                if target in alive_players and target != self.id:
                    return target

        # 4. [대립 정리] 맞경 -> 맞의 순서로 정리
        if len(police_claims) >= 2:
            target = police_claims[0]
            if target == self.id: target = police_claims[1]
            return target
        
        doctor_claims = [pid for pid, role in context["claims"].items() if role == Role.DOCTOR and pid in alive_players]
        if len(doctor_claims) >= 2:
            target = doctor_claims[0]
            if target == self.id: target = doctor_claims[1]
            return target

        return None

    def _act_discussion(self, status: GameStatus, context: dict) -> GameAction:
        alive_players = [p.id for p in status.players if p.alive]

        # 1. [직업 공개] 1일차 필수
        if status.day == 1 and not context["my_claim_done_today"]:
            if self.role != Role.CITIZEN:
                claim = self.fake_role if self.role == Role.MAFIA else self.role
                return GameAction(target_id=self.id, claim_role=claim)

        # 2. [시민 침묵] 확정된 경찰이나 의사가 있으면 시민은 입을 다문다 (NEW!)
        # 그들의 오더(투표)를 묵묵히 따르기 위함
        if self.role == Role.CITIZEN:
            confirmed = self._get_confirmed_players(alive_players, context)
            if confirmed:
                return GameAction(target_id=-1)

        # 3. [언행일치] 논리적 타겟 지목
        target = self._get_logic_target(alive_players, context)
        if target is not None and not context["my_accusation_done_today"]:
            return GameAction(target_id=target, claim_role=Role.MAFIA)
        
        return GameAction(target_id=-1)

    def _act_vote(self, status: GameStatus, context: dict, alive_others: List[int]) -> GameAction:
        alive_players = [p.id for p in status.players if p.alive]
        
        # 논리적 타겟 투표
        target = self._get_logic_target(alive_players, context)
        if target is not None:
            return GameAction(target_id=target)

        # 엔드게임(4인 이하) 혹은 정보 부족 시: 확정된 사람 제외하고 랜덤
        if len(alive_players) <= 4:
            confirmed = self._get_confirmed_players(alive_players, context)
            candidates = [p for p in alive_others if p not in confirmed]
            
            if candidates:
                return GameAction(target_id=random.choice(candidates))
            elif alive_others:
                 return GameAction(target_id=random.choice(alive_others))

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
        if self.role == Role.MAFIA:
            special_roles = [pid for pid, role in context["claims"].items() if role in [Role.POLICE, Role.DOCTOR]]
            exclude = special_roles + context["teammates"]
            silent_citizens = [t for t in alive_others if t not in exclude]
            
            if silent_citizens: return GameAction(target_id=random.choice(silent_citizens))
            
            valid = [t for t in alive_others if t not in context["teammates"]]
            if valid: return GameAction(target_id=random.choice(valid))
            return GameAction(target_id=-1)

        if self.role == Role.POLICE:
            # 1. 맞경 조사
            rivals = [pid for pid, role in context["claims"].items() if role == Role.POLICE and pid != self.id and pid in alive_others]
            if rivals: return GameAction(target_id=rivals[0])
            
            # 2. 맞의 조사
            doctor_claims = [pid for pid, role in context["claims"].items() if role == Role.DOCTOR and pid in alive_others]
            if len(doctor_claims) >= 2:
                uninvestigated_docs = [t for t in doctor_claims if t not in context["investigations"]]
                if uninvestigated_docs:
                    return GameAction(target_id=random.choice(uninvestigated_docs))

            # 3. 일반인 조사
            confirmed = self._get_confirmed_players(alive_others, context)
            uninvestigated = [t for t in alive_others if t not in context["investigations"] and t not in confirmed]
            
            if uninvestigated: return GameAction(target_id=random.choice(uninvestigated))
            return GameAction(target_id=random.choice(alive_others))

        if self.role == Role.DOCTOR:
            # [수정됨] 소거법: 진짜 경찰이 죽었는지 확인
            dead_real_police = [pid for pid, r in context["known_roles"].items() if r == Role.POLICE]
            
            # 진짜 경찰이 이미 죽었다면? -> 남은 경찰 호소인은 100% 마피아 -> 절대 보호 안 함 (자힐)
            if dead_real_police:
                return GameAction(target_id=self.id)

            # 진짜 경찰이 살아있을 가능성이 있으면 경찰 호소인 보호
            police_claims = [pid for pid, role in context["claims"].items() if role == Role.POLICE and pid in alive_others]
            if police_claims: return GameAction(target_id=random.choice(police_claims))
            
            return GameAction(target_id=self.id)

        return GameAction(target_id=-1)