from typing import Dict, List, Optional
from collections import defaultdict
from config import Role, EventType, Phase
from core.engine.state import GameEvent

import numpy as np

# === [1. Reward Parameters] ===
WIN_REWARD = 1.0
LOSS_PENALTY = -1.0
TIME_PENALTY_UNIT = -0.1
DECEPTION_MULTIPLIER = 1.5
REWARD_LAMBDA = 0.5
INT_REWARD_SCALE = 25.0

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
        self.last_claims: Dict[int, Role] = {}
        self.cumulative_intermediate = defaultdict(float)
        self.rewards = defaultdict(float)        

    def reset(self):
        """에피소드 초기화 시 호출"""
        self.last_claims.clear()

    def calculate(self, game, start_idx: int = 0) -> Dict[int, float]:
        """
        게임 히스토리를 분석하여 중간 보상을 누적하고, 
        게임 종료 시점에 최종 정규화된 보상을 계산합니다.
        """
        # 이번 호출에서 반환할 보상 (최종 계산 전까지는 0)
        step_rewards = defaultdict(float)
        game_ended = False
        citizen_win = False

        # 1. 중간 보상 누적 (시간 페널티)
        # 생존자 대상 시간 페널티를 중간 보상 합계(ΣR_int)에 누적
        daily_penalty = TIME_PENALTY_UNIT * game.day
        for p in game.players:
            if p.alive:
                self.cumulative_intermediate[p.id] += daily_penalty

        # 2. 히스토리 기반 이벤트 분석
        new_events = game.history[start_idx:]
        
        for event in new_events:
            # A. 게임 종료 이벤트 감지
            if event.phase == Phase.GAME_END:
                game_ended = True
                citizen_win = event.value
                continue

            # B. 주장(Claim) 업데이트 (기존 유지)
            if event.event_type == EventType.CLAIM and event.actor_id != -1:
                if isinstance(event.value, Role):
                    self.last_claims[event.actor_id] = event.value
                continue

            # C. 상호작용 보상 누적 (ΣR_int에 합산)
            target_role = self._get_target_role(game, event.target_id)
            
            for p in game.players:
                if not p.alive:
                    continue
                
                base_reward = REWARD_MATRIX[p.role][event.event_type][target_role]
                if base_reward == 0:
                    continue

                multiplier = 1.0
                if p.role == Role.MAFIA and base_reward > 0:
                    if self.last_claims.get(p.id) == Role.POLICE:
                        multiplier = DECEPTION_MULTIPLIER
                
                # 보상을 즉시 반환하지 않고 누적 변수에 저장
                self.cumulative_intermediate[p.id] += base_reward * multiplier

        # 3. 최종 공식 적용 (게임이 종료되었을 때만 실행)
        if game_ended:
            for p in game.players:
                # 3-1. 승패 보상 결정 (R_win)
                is_citizen_team = (p.role != Role.MAFIA)
                if citizen_win:
                    r_win = WIN_REWARD if is_citizen_team else LOSS_PENALTY
                else:
                    r_win = WIN_REWARD if not is_citizen_team else LOSS_PENALTY

                # 3-2. 공식 적용: R = λ * tanh(ΣR_int / k_int) + (1-λ) * R_win
                r_int_sum = self.cumulative_intermediate[p.id]
                r_int_norm = np.tanh(r_int_sum / INT_REWARD_SCALE)
                
                final_reward = (REWARD_LAMBDA * r_int_norm + 
                                (1 - REWARD_LAMBDA) * r_win)
                
                step_rewards[p.id] = final_reward

        return dict(step_rewards)

    def _get_target_role(self, game, target_id: Optional[int]) -> Optional[Role]:
        """타겟 ID의 유효성을 검사하고 실제 역할을 반환합니다."""
        if target_id is not None and 0 <= target_id < len(game.players):
            return game.players[target_id].role
        return None