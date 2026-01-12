import random
from typing import List
from core.agents.base_agent import BaseAgent
from core.engine.state import GameStatus, GameAction
from config import Role, Phase, ActionType


class RandonAgent(BaseAgent):
    def __init__(self, player_id: int, role: Role):
        super().__init__(player_id, role)

    def get_action(self, status: GameStatus) -> GameAction:
        alive_players = [p.id for p in status.players if p.alive and p.id != self.id]

        if status.phase == Phase.DAY_DISCUSSION:
            return self._act_discussion(status, alive_players)

        elif status.phase == Phase.DAY_VOTE:
            return self._act_vote(status, alive_players)

        elif status.phase == Phase.DAY_EXECUTE:
            return GameAction(target_id=0, claim_role=None)

        elif status.phase == Phase.NIGHT:
            return self._act_night(status, alive_players)

        return GameAction(target_id=-1, claim_role=None)

    def _act_discussion(self, status: GameStatus, targets: List[int]) -> GameAction:
        """낮 토론: 보통 침묵하거나 역할을 밝힘"""
        if random.random() < 0.1:
            return GameAction(target_id=self.id, claim_role=self.role)

        # 침묵 (PASS)
        return GameAction(target_id=-1, claim_role=None)

    def _act_vote(self, status: GameStatus, targets: List[int]) -> GameAction:
        """투표: 의심가는 사람 투표"""
        if not targets:
            return GameAction(target_id=-1)

        target = random.choice(targets)

        return GameAction(target_id=target)

    def _act_night(self, status: GameStatus, targets: List[int]) -> GameAction:
        """밤: 직업별 특수 능력 사용"""
        if not targets:
            return GameAction(target_id=-1)

        # 마피아: 살해
        if self.role == Role.MAFIA:
            target = random.choice(targets)
            return GameAction(target_id=target)

        # 경찰: 조사
        elif self.role == Role.POLICE:
            target = random.choice(targets)
            return GameAction(target_id=target)

        # 의사: 보호
        elif self.role == Role.DOCTOR:
            if random.random() < 0.5:
                return GameAction(target_id=self.id)
            else:
                return GameAction(target_id=random.choice(targets))

        # 시민: 밤에 할 일 없음
        return GameAction(target_id=-1)
