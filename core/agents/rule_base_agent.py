import random
from typing import List, Set
from core.agents.base_agent import BaseAgent
from core.engine.state import GameStatus, GameAction
from config import EventType, Role, Phase


class RuleBaseAgent(BaseAgent):
    def __init__(self, player_id: int, role: Role):
        super().__init__(player_id, role)
        self.investigated_players: Set[int] = set()  # 경찰 조사 리스트
        self.known_mafia: Set[int] = set()  # 알아낸 마피아 목록
        self.suspects: Set[int] = set()  # 의심스러운 플레이어 리스트

    def get_action(self, status):
        self.role = status.my_role
        # 의심도
        self._update_knowledge(status)

        alive_others = [p.id for p in status.players if p.alive and p.id != self.id]

        if status.phase == Phase.DAY_DISCUSSION:
            return self._act_discussion(status, alive_others)

        elif status.phase == Phase.DAY_VOTE:
            return self._act_vote(status, alive_others)

        elif status.phase == Phase.DAY_EXECUTE:
            return self._act_execute(status, alive_others)

        elif status.phase == Phase.NIGHT:
            return self._act_night(status, alive_others)

        return GameAction(target_id=-1)

    def _act_discussion(self, status: GameStatus, targets: List[int]) -> GameAction:
        # 시민은 토론에서 특별한 행동을 하지 않음
        # 마피아는 경찰이 나오면 경찰 주장
        # 의사는 특별한 행동을 하지 않음
        # 경찰은 마피아 맞추면 자신이 경찰이라고 주장 하며 마피아 조사

        # 1. 시민 & 의사: 특별한 행동 없음
        if self.role == Role.CITIZEN or self.role == Role.DOCTOR:
            # 기존 로직: 그냥 지목만 함
            if targets and random.random() < 0.8:
                return GameAction(target_id=random.choice(targets))

        # 2. 마피아
        if self.role == Role.MAFIA:
            # 맞경
            for event in status.action_history:
                # if event.day == status.day and event.event_type == EventType.CLAIM:
                #     pass

                if (
                    event.day == status.day
                    and event.phase == Phase.DAY_DISCUSSION
                    and event.event_type == EventType.CLAIM
                    and event.value == Role.MAFIA
                    and event.target_id == self.id
                    and event.actor_id != self.id
                ):
                    return GameAction(target_id=event.actor_id, claim_role=Role.POLICE)

            # 시민 모방
            if targets and random.random() < 0.8:
                return GameAction(target_id=random.choice(targets))

            # 평소에는 시민과 똑같이 행동
            if targets and random.random() < 0.8:
                return GameAction(target_id=random.choice(targets))

        # 3. 경찰
        if self.role == Role.POLICE:
            alive_mafia = [m for m in self.known_mafia if m in targets]

            if alive_mafia:
                target = alive_mafia[0]
                return GameAction(target_id=target, claim_role=Role.MAFIA)

            if targets and random.random() < 0.8:
                return GameAction(target_id=random.choice(targets))

        return GameAction(target_id=-1)

    def _act_vote(self, status: GameStatus, targets: List[int]) -> GameAction:
        if not targets:
            return GameAction(target_id=-1)

        # 1. 오늘 경찰이라고 주장한 사람들을 모두 수집
        police_claims = {}

        for event in status.action_history:
            if (
                event.day == status.day
                and event.phase == Phase.DAY_DISCUSSION
                and event.event_type == EventType.CLAIM
                and event.value == Role.MAFIA
            ):

                # {경찰 주장자 ID : 그가 지목한 마피아 후보 ID}
                police_claims[event.actor_id] = event.target_id

        # [1] 시민 & 의사
        if self.role == Role.CITIZEN or self.role == Role.DOCTOR:
            # 맞경
            if len(police_claims) >= 2:
                claimants = [pid for pid in police_claims.keys() if pid in targets]
                if claimants:
                    return GameAction(target_id=random.choice(claimants))

            # 경찰이 1명
            elif len(police_claims) == 1:
                police_target = list(police_claims.values())[0]
                if police_target != -1 and police_target in targets:
                    return GameAction(target_id=police_target)

            # 소신 투표
            valid_suspects = [s for s in self.suspects if s in targets]
            if valid_suspects:
                return GameAction(target_id=random.choice(valid_suspects))

            return GameAction(target_id=random.choice(targets))

        # [2] 마피아
        elif self.role == Role.MAFIA:
            # 나(self.id)를 제외한 경찰 주장자 목록
            enemy_claimants = [
                pid for pid in police_claims.keys() if pid in targets and pid != self.id
            ]

            if enemy_claimants:
                return GameAction(target_id=random.choice(enemy_claimants))

            # 경찰 없으면 의심자나 아무나 투표
            valid_suspects = [s for s in self.suspects if s in targets]
            if valid_suspects:
                return GameAction(target_id=random.choice(valid_suspects))
            return GameAction(target_id=random.choice(targets))

        # [3] 경찰
        elif self.role == Role.POLICE:
            # 마피아
            alive_mafia = [m for m in self.known_mafia if m in targets]
            if alive_mafia:
                return GameAction(target_id=alive_mafia[0])

            # 맞경한 대상
            fake_police = [
                pid for pid in police_claims.keys() if pid in targets and pid != self.id
            ]
            if fake_police:
                return GameAction(target_id=fake_police[0])

            # 의심자 투표
            valid_suspects = [s for s in self.suspects if s in targets]
            if valid_suspects:
                return GameAction(target_id=random.choice(valid_suspects))
            return GameAction(target_id=random.choice(targets))

        return GameAction(target_id=-1)

    def _act_execute(self, status: GameStatus, targets: List[int]) -> GameAction:
        # 무조건 찬성
        # 1. 오늘 낮 투표 결과
        vote_counts = {}
        for event in status.action_history:
            if event.day == status.day and event.phase == Phase.DAY_VOTE:
                if event.target_id != -1:
                    vote_counts[event.target_id] = (
                        vote_counts.get(event.target_id, 0) + 1
                    )

        # 2. 최다 득표자(처형 후보) 찾기
        if not vote_counts:
            return GameAction(target_id=-1)

        candidate = max(vote_counts, key=vote_counts.get)

        # 3. 처형 후보에게 찬성표(지목) 행사
        if candidate in targets:
            return GameAction(target_id=candidate)

        return GameAction(target_id=-1)

    def _act_night(self, status: GameStatus, targets: List[int]) -> GameAction:
        # 시민은 행동 없음
        # 마피아는 시민 중 무작위 살해
        # 의사는 의심스러운 플레이어 중 무작위 보호
        # 경찰은 의심스러운 플레이어 중 무작위 조사

        # 1. 시민: 행동 없음
        if self.role == Role.CITIZEN:
            return GameAction(target_id=-1)

        # 2. 마피아: 시민 중 무작위 살해
        if self.role == Role.MAFIA:
            if not targets:
                return GameAction(target_id=-1)

            return GameAction(target_id=random.choice(targets))

        # 3. 의사: 자신 보호
        if self.role == Role.DOCTOR:
            return GameAction(target_id=self.id)

        # 4. 경찰: 의심스러운 플레이어 중 무작위 조사
        if self.role == Role.POLICE:
            # 의심가는 사람 중 + 아직 조사 안 한 사람
            priority_targets = [
                t
                for t in self.suspects
                if t in targets and t not in self.investigated_players
            ]

            if priority_targets:
                target = random.choice(priority_targets)
                self.investigated_players.add(target)
                return GameAction(target_id=target)

            # 조사 안 한 나머지 생존자 중 랜덤
            unknowns = [t for t in targets if t not in self.investigated_players]

            if unknowns:
                target = random.choice(unknowns)
            elif targets:
                target = random.choice(targets)
            else:
                return GameAction(target_id=-1)

            # 조사 목록 업데이트
            self.investigated_players.add(target)
            return GameAction(target_id=target)

        return GameAction(target_id=-1)

    def _update_knowledge(self, status: GameStatus):
        # 1. [공통] 나에게 투표한 사람을 의심 목록에 추가 (복수 심리)
        for event in status.action_history:
            # 누군가 투표(VOTE) 단계에서 나(self.id)를 찍었다면
            if (
                event.phase == Phase.DAY_VOTE
                and event.target_id == self.id
                and event.actor_id != self.id
            ):

                self.suspects.add(event.actor_id)

        # 2. [경찰] 조사 결과 확인 및 마피아 판별
        if self.role == Role.POLICE:
            for event in status.action_history:
                if (
                    event.event_type == EventType.POLICE_RESULT
                    and event.actor_id == self.id
                ):

                    target = event.target_id
                    if target is not None:
                        # 조사한 사람 추가
                        self.investigated_players.add(target)

                        # 마피아 판단
                        if event.value == Role.MAFIA:
                            self.known_mafia.add(target)
