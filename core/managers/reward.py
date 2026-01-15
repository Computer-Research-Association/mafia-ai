from typing import Dict, List, Optional
from collections import defaultdict
from config import Role, EventType, Phase
from core.engine.state import GameEvent

# === [1. Reward Parameters] ===
WIN_REWARD = 10.0
LOSS_PENALTY = -10.0
TIME_PENALTY_UNIT = -0.1
DECEPTION_MULTIPLIER = 1.5

# === [2. Reward Matrix Setup] ===
# [내 역할][이벤트 타입][대상 역할] 구조의 3차원 행렬
REWARD_MATRIX = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

# --- MAFIA Team Rewards ---
_M = Role.MAFIA
REWARD_MATRIX[_M][EventType.EXECUTE][Role.POLICE] = 10.0
REWARD_MATRIX[_M][EventType.EXECUTE][Role.DOCTOR] = 10.0
REWARD_MATRIX[_M][EventType.EXECUTE][Role.CITIZEN] = 3.0
REWARD_MATRIX[_M][EventType.EXECUTE][Role.MAFIA] = -20.0

REWARD_MATRIX[_M][EventType.KILL][Role.POLICE] = 5.0
REWARD_MATRIX[_M][EventType.KILL][Role.DOCTOR] = 5.0
REWARD_MATRIX[_M][EventType.KILL][Role.CITIZEN] = 2.0
REWARD_MATRIX[_M][EventType.KILL][Role.MAFIA] = -99.0

# --- CITIZEN Team Rewards ---
CIV_ROLES = [Role.CITIZEN, Role.POLICE, Role.DOCTOR]
for r in CIV_ROLES:
    # 처형 보상/페널티
    REWARD_MATRIX[r][EventType.EXECUTE][Role.MAFIA] = 10.0
    REWARD_MATRIX[r][EventType.EXECUTE][Role.CITIZEN] = -5.0
    REWARD_MATRIX[r][EventType.EXECUTE][Role.POLICE] = -10.0
    REWARD_MATRIX[r][EventType.EXECUTE][Role.DOCTOR] = -10.0
    
    # 사망 페널티 (팀원 손실)
    REWARD_MATRIX[r][EventType.KILL][Role.CITIZEN] = -1.0
    REWARD_MATRIX[r][EventType.KILL][Role.POLICE] = -2.0
    REWARD_MATRIX[r][EventType.KILL][Role.DOCTOR] = -2.0

# --- Role Specific Interactions ---
REWARD_MATRIX[Role.POLICE][EventType.POLICE_RESULT][Role.MAFIA] = 3.0
REWARD_MATRIX[Role.POLICE][EventType.POLICE_RESULT][Role.CITIZEN] = 0.5
REWARD_MATRIX[Role.DOCTOR][EventType.PROTECT][Role.POLICE] = 1.0
REWARD_MATRIX[Role.DOCTOR][EventType.PROTECT][Role.CITIZEN] = 0.5


class RewardManager:
    def __init__(self):
        # 기만 보너스 계산을 위한 플레이어별 마지막 주장(Claim) 기록
        self.last_claims: Dict[int, Role] = {}

    def reset(self):
        """에피소드 초기화 시 호출"""
        self.last_claims.clear()

    def calculate(self, game, start_idx: int = 0) -> Dict[int, float]:
        """
        게임 히스토리를 분석하여 모든 플레이어의 보상을 일괄 계산합니다.
        """
        rewards = defaultdict(float)
        
        # 1. 시간 페널티 적용 (생존자 대상)
        daily_penalty = TIME_PENALTY_UNIT * game.day
        for p in game.players:
            if p.alive:
                rewards[p.id] += daily_penalty

        # 2. 히스토리 기반 이벤트 분석
        new_events = game.history[start_idx:]
        
        for event in new_events:
            # A. 게임 종료 처리 (Phase.GAME_END와 EventType.EXECUTE 값 충돌 방지)
            if event.phase == Phase.GAME_END:
                citizen_win = event.value
                for p in game.players:
                    is_citizen_team = (p.role != Role.MAFIA)
                    # 승리 팀에게는 WIN_REWARD, 패배 팀에게는 LOSS_PENALTY 부여
                    if citizen_win:
                        rewards[p.id] += WIN_REWARD if is_citizen_team else LOSS_PENALTY
                    else:
                        rewards[p.id] += WIN_REWARD if not is_citizen_team else LOSS_PENALTY
                continue

            # B. 주장(Claim) 업데이트 (사칭 상태 추적)
            if event.event_type == EventType.CLAIM and event.actor_id != -1:
                if isinstance(event.value, Role):
                    self.last_claims[event.actor_id] = event.value
                continue

            # C. 상호작용 보상 계산 (Matrix Lookup)
            target_role = self._get_target_role(game, event.target_id)
            
            for p in game.players:
                if not p.alive:
                    continue
                
                # 행렬에서 기본 보상 조회
                base_reward = REWARD_MATRIX[p.role][event.event_type][target_role]
                if base_reward == 0:
                    continue

                # 마피아 기만 보너스 (경찰 사칭 중 득점 시 적용)
                multiplier = 1.0
                if p.role == Role.MAFIA and base_reward > 0:
                    if self.last_claims.get(p.id) == Role.POLICE:
                        multiplier = DECEPTION_MULTIPLIER
                
                rewards[p.id] += base_reward * multiplier

        return dict(rewards)

    def _get_target_role(self, game, target_id: Optional[int]) -> Optional[Role]:
        """타겟 ID의 유효성을 검사하고 실제 역할을 반환합니다."""
        if target_id is not None and 0 <= target_id < len(game.players):
            return game.players[target_id].role
        return None