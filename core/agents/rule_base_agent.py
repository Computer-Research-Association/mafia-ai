import random
from typing import List, Dict, Optional
from core.agents.base_agent import BaseAgent
from core.engine.state import GameStatus, GameAction
from config import EventType, Role, Phase

class RuleBaseAgent(BaseAgent):
    def __init__(self, player_id: int, role: Role):
        super().__init__(player_id, role)
        self.reset()

    def reset(self):
        # 마피아는 게임 시작 시 위장 직업(경찰 or 의사)을 하나 정함
        self.fake_role = random.choice([Role.POLICE, Role.DOCTOR]) if self.role == Role.MAFIA else None
        
        # 게임 상태 추적용 변수들
        self.claims: Dict[int, Role] = {}  # {player_id: claimed_role}
        self.investigations: Dict[int, Role] = {} # {target_id: result_role} (경찰용)
        self.known_roles: Dict[int, Role] = {} # {player_id: confirmed_role} (죽은 사람 or 경찰 정보)
        self.my_claims_made = False
        self.last_processed_idx = 0

    def get_action(self, status: GameStatus) -> GameAction:
        self.role = status.my_role
        self.id = status.my_id
        self._sync_state(status)

        # 죽은 자는 말이 없다
        if not any(p.id == self.id and p.alive for p in status.players):
            return GameAction(target_id=-1)

        alive_others = [p.id for p in status.players if p.alive and p.id != self.id]
        if not alive_others: return GameAction(target_id=-1)

        if status.phase == Phase.DAY_DISCUSSION: return self._act_discussion(status)
        if status.phase == Phase.DAY_VOTE:       return self._act_vote(status)
        if status.phase == Phase.DAY_EXECUTE:    return self._act_execute(status)
        if status.phase == Phase.NIGHT:          return self._act_night(status, alive_others)
        
        return GameAction(target_id=-1)

    def _sync_state(self, status: GameStatus):
        """게임 기록을 읽어 누가 뭘 주장했고, 누가 죽어서 정체가 뭔지 파악함"""
        events = status.action_history[self.last_processed_idx:]
        for event in events:
            # 1. 직업 공개(Claim) 추적
            if event.event_type == EventType.CLAIM and event.value:
                self.claims[event.actor_id] = event.value
            
            # 2. 처형/사망 결과로 정체 확정
            if event.event_type == EventType.EXECUTE:
                self.known_roles[event.target_id] = event.value # 죽은 자의 직업 공개
            
            # 3. (경찰만) 자신의 조사 결과 저장
            if event.event_type == EventType.POLICE_RESULT and event.actor_id == self.id:
                self.investigations[event.target_id] = event.value
                # 내가 진짜 경찰이면, 내 조사 결과는 나에게 '확정' 정보임
                if self.role == Role.POLICE:
                    self.known_roles[event.target_id] = event.value

        self.last_processed_idx = len(status.action_history)

    def _act_discussion(self, status: GameStatus) -> GameAction:
        """
        [전략 1] 1일차 등 초반에 직업 공개 (마피아는 위장, 특직은 진실, 시민은 시민)
        """
        # 아직 직업 공개를 안 했다면?
        if not self.my_claims_made:
            claim_role = self.fake_role if self.role == Role.MAFIA else self.role
            # 전략적 선택: 시민도 그냥 시민이라고 까버림 (침묵 방지)
            self.my_claims_made = True
            return GameAction(target_id=self.id, claim_role=claim_role)

        # [전략 4] 경찰(또는 짭경)은 조사 결과가 있으면 발표
        if self.role == Role.POLICE or (self.role == Role.MAFIA and self.fake_role == Role.POLICE):
            # 마피아는 아무나 찍어서 마피아라고 거짓말하거나 시민이라고 거짓말 (여기선 단순하게 생략하거나 랜덤)
            # 룰베이스 봇은 진짜 경찰 로직만 구현:
            if self.role == Role.POLICE:
                for target, result in self.investigations.items():
                    # 아직 이 사실을 공표 안 했으면? (로그 뒤져봐야 하지만 편의상 매번 말해도 됨, 엔진이 중복제거 해주길 기대하거나)
                    # 여기서는 가장 최근 조사 결과를 마피아인 경우에만 강력하게 주장
                    if result == Role.MAFIA:
                        return GameAction(target_id=target, claim_role=Role.MAFIA)
        
        return GameAction(target_id=-1)

    def _act_vote(self, status: GameStatus) -> GameAction:
        """
        [전략 2, 4, 5] 투표 로직의 핵심
        """
        alive_players = [p.id for p in status.players if p.alive]
        
        # 1. '맞경(Police Counter-claim)' 그룹 찾기
        police_claims = [pid for pid, role in self.claims.items() if role == Role.POLICE and pid in alive_players]
        
        # 2. 확정된 마피아(죽은 경찰의 대립자 or 내 조사결과)가 있으면 1순위 척살
        # 2-1. 어제 죽은 사람이 '진짜 경찰'이었다 -> 살아있는 맞경은 '마피아'다
        dead_polices = [pid for pid, role in self.known_roles.items() if role == Role.POLICE]
        for dead_cop in dead_polices:
            # 죽은 경찰과 대립했던 살아있는 경찰 호소인 찾기
            # (단순화: 경찰을 사칭했던 놈들 중 살아있는 놈은 이제 확정 마피아임)
            # 주의: 과거의 모든 police_claims 기록을 뒤져야 함. self.claims는 누적됨.
            rivals = [pid for pid, role in self.claims.items() if role == Role.POLICE and pid != dead_cop and pid in alive_players]
            if rivals:
                return GameAction(target_id=rivals[0]) # 그 놈 죽여

        # 2-2. 내가 경찰인데 조사했더니 마피아더라 -> 죽여
        if self.role == Role.POLICE:
            for target, result in self.investigations.items():
                if result == Role.MAFIA and target in alive_players:
                    return GameAction(target_id=target)

        # 3. [전략 2] 1일차: 맞경 중 하나 죽이기 (무조건)
        # 맞경이 2명 이상이면, 그냥 리스트의 첫 번째 놈을 다같이 찍어서 죽임 (합심)
        if len(police_claims) >= 2:
            # 마피아 팀은 진짜 경찰을 찍으려 노력하겠지만, 시민 룰베이스는 그냥 '첫번째 놈'을 찍자
            # (어차피 내일 결과 보고 나머지 죽이면 되니까)
            target = police_claims[0]
            if target == self.id: # 나를 찍을 순 없으니
                target = police_claims[1]
            return GameAction(target_id=target)

        # 4. 맞의(Doctor Counter-claim) 처리 (경찰 정리됐으면 의사 정리)
        doctor_claims = [pid for pid, role in self.claims.items() if role == Role.DOCTOR and pid in alive_players]
        if len(doctor_claims) >= 2:
            target = doctor_claims[0]
            if target == self.id: target = doctor_claims[1]
            return GameAction(target_id=target)

        # 5. 할 거 없으면? (확정 정보도 없고 대립도 없음)
        # 그냥 아무나 찍지 말고 기권하거나 랜덤 (여기선 랜덤 투표로 진행시킴)
        return GameAction(target_id=random.choice(alive_players))

    def _act_execute(self, status: GameStatus) -> GameAction:
        """투표 결과가 나왔을 때 찬반 -> 무조건 찬성해서 죽여야 게임이 진행됨"""
        # 현재 투표 1등 확인
        votes = {}
        for e in status.action_history:
            if e.day == status.day and e.phase == Phase.DAY_VOTE and e.event_type == EventType.VOTE:
                if e.target_id not in [None, -1]: votes[e.target_id] = votes.get(e.target_id, 0) + 1
        
        if not votes: return GameAction(target_id=self.id) # 투표 없으면 자투(반대)
        
        target_id = max(votes, key=votes.get)
        
        # 내가 걸렸거나, 우리 편(마피아끼리)이 걸렸으면 반대
        if target_id == self.id: return GameAction(target_id=self.id)
        if self.role == Role.MAFIA and target_id in self.known_roles: # known_roles엔 마피아 팀원 정보가 없으므로 별도 로직 필요하지만 생략
             pass 

        # [전략] 시민은 무조건 찬성하여 '한 놈 보내고 확인'하는 전략 수행
        return GameAction(target_id=target_id)

    def _act_night(self, status: GameStatus, targets: List[int]) -> GameAction:
        """
        [전략 3] 밤 행동 수칙
        """
        # 1. 마피아: 맞경/맞의가 '아닌' 애를 죽여서 혼란 유지 (혹은 그냥 경찰 죽이기)
        if self.role == Role.MAFIA:
            # 경찰/의사라고 주장한 놈들 제외
            special_claims = [pid for pid, role in self.claims.items() if role in [Role.POLICE, Role.DOCTOR]]
            # 제외하고 남은 '그냥 시민'들 중에서 타겟 선정
            valid_targets = [t for t in targets if t not in special_claims and t != self.id]
            
            if not valid_targets: # 다 죽었으면 아무나
                valid_targets = [t for t in targets if t != self.id]
            
            return GameAction(target_id=random.choice(valid_targets))

        # 2. 경찰: 맞경 조사 1순위, 없으면 랜덤
        if self.role == Role.POLICE:
            police_claims = [pid for pid, role in self.claims.items() if role == Role.POLICE and pid != self.id and pid in targets]
            if police_claims:
                return GameAction(target_id=police_claims[0]) # 대립자 조사
            
            # 대립자 없으면, 아직 조사 안 한 사람 중 랜덤
            uninvestigated = [t for t in targets if t not in self.investigations]
            if uninvestigated:
                return GameAction(target_id=random.choice(uninvestigated))
            return GameAction(target_id=random.choice(targets))

        # 3. 의사: 경찰 살리기 (없으면 나)
        if self.role == Role.DOCTOR:
            police_claims = [pid for pid, role in self.claims.items() if role == Role.POLICE and pid in targets]
            if police_claims:
                # 경찰이 2명이면? 도박 (반반) 혹은 그냥 첫번째
                return GameAction(target_id=random.choice(police_claims))
            return GameAction(target_id=self.id) # 경찰 없으면 자힐

        return GameAction(target_id=-1)