from typing import Dict, List, Optional
from collections import defaultdict
from config import Role, EventType, Phase
from core.engine.state import GameEvent

import os
import numpy as np

# === [1. Reward Parameters] ===
WIN_REWARD = 1.0
LOSS_PENALTY = -1.0
# INT_REWARD_SCALE: 즉각 보상을 스케일링하기 위한 상수
# 게임당 평균 10~15개 이벤트를 고려하여 중간/최종 보상 균형
# 람다 0.5 기준: 개별 이벤트 0.5, 누적 후 최종 보상과 비슷한 스케일
INT_REWARD_SCALE = 10.0
CORRECT_KEY = "CORRECT_ACCUSE"

# === [2. Reward Matrix Setup] ===
# [내 역할][이벤트 타입][대상 역할] 구조의 3차원 행렬
# 핵심 이벤트만 유지: EXECUTE, KILL, CORRECT_KEY
REWARD_MATRIX = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

# --- MAFIA Team Rewards ---
_M = Role.MAFIA
_C = Role.CITIZEN
_D = Role.DOCTOR
_P = Role.POLICE

# EXECUTE: 처형 결과에 대한 보상
REWARD_MATRIX[_M][EventType.EXECUTE][_P] = 10.0
REWARD_MATRIX[_M][EventType.EXECUTE][_D] = 10.0
REWARD_MATRIX[_M][EventType.EXECUTE][_C] = 3.0
REWARD_MATRIX[_M][EventType.EXECUTE][_M] = -20.0

# KILL: 마피아의 살해 행동
REWARD_MATRIX[_M][EventType.KILL][_P] = 5.0
REWARD_MATRIX[_M][EventType.KILL][_D] = 5.0
REWARD_MATRIX[_M][EventType.KILL][_C] = 2.0
REWARD_MATRIX[_M][EventType.KILL][_M] = -99.0

# CORRECT_KEY: 정확한 지목 (추론 능력)
REWARD_MATRIX[_M][CORRECT_KEY][_M] = -5.0
REWARD_MATRIX[_M][CORRECT_KEY][_P] = -2.0
REWARD_MATRIX[_M][CORRECT_KEY][_D] = -1.0
REWARD_MATRIX[_M][CORRECT_KEY][_C] = -0.5

# --- CIVILIAN Team Rewards ---
CIV_ROLES = [_C, _P, _D]
for r in CIV_ROLES:
    # CORRECT_KEY: 정확한 지목 보상
    REWARD_MATRIX[r][CORRECT_KEY][r] = 3.0
    REWARD_MATRIX[r][CORRECT_KEY][_M] = 5.0

    # EXECUTE: 처형 보상/페널티
    REWARD_MATRIX[r][EventType.EXECUTE][_M] = 10.0
    REWARD_MATRIX[r][EventType.EXECUTE][_C] = -5.0
    REWARD_MATRIX[r][EventType.EXECUTE][_P] = -10.0
    REWARD_MATRIX[r][EventType.EXECUTE][_D] = -10.0

    # KILL: 사망 페널티 (팀원 손실)
    REWARD_MATRIX[r][EventType.KILL][_C] = -1.0
    REWARD_MATRIX[r][EventType.KILL][_P] = -2.0
    REWARD_MATRIX[r][EventType.KILL][_D] = -2.0

# --- Role Specific Interactions ---
# 역할 특화 보상: 역할별 학습에 중요한 신호
REWARD_MATRIX[_P][EventType.POLICE_RESULT][_M] = 3.0
REWARD_MATRIX[_P][EventType.POLICE_RESULT][_C] = 0.5
REWARD_MATRIX[_D][EventType.PROTECT][_P] = 1.0
REWARD_MATRIX[_D][EventType.PROTECT][_C] = 0.5


class RewardManager:
    def __init__(self):
        self.last_claims: Dict[int, Role] = {}
        self.reward_lambda = float(os.getenv("MAFIA_LAMBDA"))

    def reset(self):
        """에피소드 초기화 시 호출"""
        self.last_claims.clear()

    def calculate(self, game, start_idx: int = 0) -> Dict[int, float]:
        """
        게임 히스토리를 분석하여 매 스텝 즉각적인 보상을 반환합니다.
        
        중간 보상: λ * (base_reward / INT_REWARD_SCALE) 형태로 즉시 반환
        게임 종료 시: (1 - λ) * R_win 형태로 승리/패배 보상 추가
        """
        step_rewards = defaultdict(float)
        game_ended = False
        citizen_win = False

        # 히스토리 기반 이벤트 분석
        new_events = game.history[start_idx:]

        for event in new_events:
            # A. 게임 종료 이벤트 감지
            if event.phase == Phase.GAME_END:
                game_ended = True
                citizen_win = event.value
                continue

            # B. 주장(Claim) 업데이트 - CORRECT_KEY 보상을 위해 유지
            if event.event_type == EventType.CLAIM and event.actor_id != -1:
                actor = game.players[event.actor_id]
                is_self_claim = event.target_id == -1 or event.target_id == actor.id

                reward_target_role = None
                lookup_event_type = None

                if is_self_claim:
                    # 자기 역할 주장은 보상 없음 (단순히 기록만)
                    if isinstance(event.value, Role):
                        self.last_claims[actor.id] = event.value
                else:
                    # 타인 지목: 정확한지 확인
                    target_real_role = self._get_target_role(game, event.target_id)
                    claimed_role = event.value

                    if (
                        isinstance(claimed_role, Role)
                        and target_real_role == claimed_role
                    ):
                        lookup_event_type = CORRECT_KEY
                        reward_target_role = target_real_role

                # 즉각 보상 반환
                if reward_target_role is not None and lookup_event_type is not None:
                    base_reward = REWARD_MATRIX[actor.role][lookup_event_type][
                        reward_target_role
                    ]
                    if base_reward != 0:
                        # 람다 가중치를 적용하여 스케일링 후 즉시 반환
                        step_rewards[actor.id] += self.reward_lambda * (base_reward / INT_REWARD_SCALE)
                continue

            # C. 상호작용 보상 즉시 반환 (EXECUTE, KILL만 처리)
            target_role = self._get_target_role(game, event.target_id)
            if target_role is None:
                continue

            for p in game.players:
                if not p.alive:
                    continue

                base_reward = REWARD_MATRIX[p.role][event.event_type][target_role]
                if base_reward == 0:
                    continue

                # 람다 가중치를 적용하여 스케일링 후 즉시 반환
                step_rewards[p.id] += self.reward_lambda * (base_reward / INT_REWARD_SCALE)

        # D. 게임 종료 시 승리/패배 보상 추가
        if game_ended:
            for p in game.players:
                # 승패 보상 결정 (R_win)
                is_citizen_team = p.role != Role.MAFIA
                if citizen_win:
                    r_win = WIN_REWARD if is_citizen_team else LOSS_PENALTY
                else:
                    r_win = WIN_REWARD if not is_citizen_team else LOSS_PENALTY

                # 최종 보상: (1-λ) * R_win
                # 중간보상은 이미 매 스텝 지급되었으므로 승리보상만 추가
                final_reward = (1 - self.reward_lambda) * r_win
                step_rewards[p.id] += final_reward

        return dict(step_rewards)

    def _get_target_role(self, game, target_id: Optional[int]) -> Optional[Role]:
        """타겟 ID의 유효성을 검사하고 실제 역할을 반환합니다."""
        if target_id is not None and 0 <= target_id < len(game.players):
            return game.players[target_id].role
        return None
