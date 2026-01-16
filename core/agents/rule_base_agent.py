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
        self.mafia_teammates: Set[int] = set() # 마피아 팀원 전용 리스트
        self.investigated_players: Set[int] = set()

    def reset(self):
        self.suspicion_scores.clear()
        self.attention_scores.clear()
        self.role_claims.clear()
        self.proven_citizens = {self.id}
        self.known_mafia.clear()
        self.mafia_teammates.clear()
        self.investigated_players.clear()

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
        # 매번 초기화하여 중복 점수 합산 방지
        self.suspicion_scores = {p.id: 10.0 for p in status.players}
        self.attention_scores = {p.id: 10.0 for p in status.players}
        self.role_claims = {}
        self.proven_citizens = {self.id}
        self.known_mafia = set()
        # 마피아 팀원 정보는 초기화하지 않음 (게임 시작 시 한 번 들어옴)

        for event in status.action_history:
            # 1. 역할 주장 및 대립 감지 (None/침묵 제외)
            if event.event_type == EventType.CLAIM:
                self.role_claims[event.actor_id] = event.value
                self.attention_scores[event.actor_id] += 5.0
                
                # 침묵(None)이나 시민 주장은 대립에서 제외
                if event.value is not None and event.value != Role.CITIZEN:
                    rivals = [pid for pid, role in self.role_claims.items() 
                              if role == event.value and pid != event.actor_id]
                    if rivals:
                        for r_id in rivals: self.suspicion_scores[r_id] += 20.0
                        self.suspicion_scores[event.actor_id] += 20.0

            # 2. 투표 기록 반영
            if event.event_type == EventType.VOTE:
                if event.target_id == self.id:
                    self.suspicion_scores[event.actor_id] += 15.0

            # 3. 확정 정보 처리 (마피아 팀킬 방지 로직 포함)
            if event.event_type == EventType.POLICE_RESULT:
                is_reliable = (self.role == Role.POLICE and event.actor_id == self.id) or \
                              (self.role == Role.MAFIA and event.actor_id == -1)
                
                if is_reliable:
                    if event.value == Role.MAFIA:
                        if self.role == Role.MAFIA:
                            # 마피아는 팀원을 별도 관리하고 공격 대상에서 절대 제외
                            self.mafia_teammates.add(event.target_id)
                            self.suspicion_scores[event.target_id] = 0.0
                        else:
                            self.known_mafia.add(event.target_id)
                            self.suspicion_scores[event.target_id] = 100.0
                    else:
                        self.proven_citizens.add(event.target_id)
                        self.suspicion_scores[event.target_id] = 0.0

    def _act_discussion(self, status: GameStatus, targets: List[int]) -> GameAction:
        if self.role == Role.POLICE:
            for m_id in self.known_mafia:
                if m_id in targets:
                    # "저놈이 마피아다"라고 정확히 주장 (Role.MAFIA)
                    return GameAction(target_id=m_id, claim_role=Role.MAFIA)
        
        if self.role == Role.MAFIA:
            if self.suspicion_scores.get(self.id, 0) > 60:
                # 위기 시 선량한 시민 한 명을 마피아로 몰기
                potential = [t for t in targets if t not in self.mafia_teammates]
                if potential:
                    target = random.choice(potential)
                    return GameAction(target_id=target, claim_role=Role.MAFIA)

        return GameAction(target_id=-1)

    def _act_vote(self, status: GameStatus, targets: List[int]) -> GameAction:
        if not targets: return GameAction(target_id=-1)

        # 마피아는 팀원을 절대 투표하지 않음
        valid_targets = [t for t in targets if t not in self.mafia_teammates] if self.role == Role.MAFIA else targets
        if not valid_targets: return GameAction(target_id=-1)

        # 시민 팀은 확정 마피아 우선 투표
        if self.role != Role.MAFIA:
            for m_id in self.known_mafia:
                if m_id in valid_targets: return GameAction(target_id=m_id)

        # 점수 계산 시 랜덤 노이즈 추가 (0/1번 몰표 방지)
        best_target = -1
        max_score = -9999.0
        for tid in valid_targets:
            score = self.suspicion_scores.get(tid, 10.0) * 1.5
            score += self.attention_scores.get(tid, 10.0) * 0.5
            score += random.uniform(-10, 30) # 랜덤 가중치
            
            if tid in self.proven_citizens: score -= 50.0 # 확정 시민 보호

            if score > max_score:
                max_score = score
                best_target = tid

        return GameAction(target_id=best_target)

    def _act_execute(self, status: GameStatus, targets: List[int]) -> GameAction:
        # 투표 결과 기반 후보 추론
        vote_counts = {}
        for event in status.action_history:
            if event.day == status.day and event.phase == Phase.DAY_VOTE and event.event_type == EventType.VOTE:
                tid = event.target_id
                if tid != -1: vote_counts[tid] = vote_counts.get(tid, 0) + 1

        candidate_id = -1
        if vote_counts:
            max_votes = max(vote_counts.values())
            top_targets = [tid for tid, count in vote_counts.items() if count == max_votes]
            if len(top_targets) == 1: candidate_id = top_targets[0]

        if candidate_id != -1:
            # 첫날 무지성 처형 방지 (상대적 의심도 체크)
            target_sus = self.suspicion_scores.get(candidate_id, 10.0)
            avg_sus = sum(self.suspicion_scores.values()) / len(self.suspicion_scores)
            
            chance = 0.55 + random.uniform(-0.15, 0.15)
            if target_sus <= avg_sus: chance -= 0.1 # 근거 부족 시 찬성 확률 감소
            if candidate_id in self.proven_citizens or candidate_id in self.mafia_teammates: chance = 0.0

            if chance > 0.5: return GameAction(target_id=candidate_id)
        return GameAction(target_id=-1)

    def _act_night(self, status: GameStatus, targets: List[int]) -> GameAction:
        if self.role == Role.CITIZEN: return GameAction(target_id=-1)

        if self.role == Role.MAFIA:
            # 팀원 제외 (팀킬 방지)
            potential = [t for t in targets if t not in self.mafia_teammates]
            if not potential: return GameAction(target_id=-1)
            # 주목도는 높고 의심도는 낮은 시민 우선 타겟
            target = max(potential, key=lambda x: self.attention_scores.get(x, 0) - self.suspicion_scores.get(x, 0) + random.uniform(-5, 5))
            return GameAction(target_id=target)

        if self.role == Role.DOCTOR:
            # 경찰 주장자 우선 보호
            police_claims = [pid for pid, role in self.role_claims.items() if role == Role.POLICE and pid in targets]
            if police_claims: return GameAction(target_id=police_claims[0])
            return GameAction(target_id=self.id)

        if self.role == Role.POLICE:
            unknowns = [t for t in targets if t not in self.investigated_players]
            if not unknowns: return GameAction(target_id=-1)
            target = max(unknowns, key=lambda x: self.suspicion_scores.get(x, 10) + random.uniform(-5, 5))
            self.investigated_players.add(target)
            return GameAction(target_id=target)

        return GameAction(target_id=-1)