# 역할 지정
ROLE_CITIZEN = 0
ROLE_POLICE = 1
ROLE_DOCTOR = 2
ROLE_MAFIA = 3

# 게임 단계 (PHASE_DAY_CLAIM removed - merged with PHASE_DAY_DISCUSSION)
PHASE_DAY_DISCUSSION = "day_discussion"  # 낮: 토론 및 주장 단계
PHASE_DAY_VOTE = "day_vote"  # 낮: 투표 단계
PHASE_DAY_EXECUTE = "day_execute"  # 낮: 처형 단계
PHASE_NIGHT = "night"  # 밤: 행동 단계

# 게임 설정
PLAYER_COUNT = 8
MAX_DAYS = 20  # 최대 턴 수 (무승부 조건)
ROLES = [
    ROLE_MAFIA,
    ROLE_MAFIA,
    ROLE_POLICE,
    ROLE_DOCTOR,
    ROLE_CITIZEN,
    ROLE_CITIZEN,
    ROLE_CITIZEN,
    ROLE_CITIZEN,
]

# === [학습 설정 최적화 v2.0] ===
# 152차원의 복잡한 관측 공간 + 21개 액션 공간에 최적화된 하이퍼파라미터

# 기본 학습 파라미터
LR = 0.0001  # Learning Rate (안정적인 학습)
GAMMA = 0.99  # Discount Factor (장기 보상 고려)
EPS_CLIP = 0.2  # PPO Clip range (정책 업데이트 제한)
K_EPOCHS = 4  # Update epochs (중간 값 유지)

# === [배치 크기 대폭 증가] ===
# 복잡한 관측 공간(152차원)과 확장된 액션 공간(21개)에 대응
# 더 많은 샘플로 안정적인 gradient 계산
BATCH_SIZE = 256  # 64 → 256으로 대폭 증가 (4배)

# === [탐험-활용 균형 최적화] ===
# Entropy Coefficient 증가: 초기 학습 시 충분한 탐험 유도
# 50% 부근 정체 해결을 위해 다양한 전략 시도 필요
ENTROPY_COEF = 0.05  # 0.01 → 0.05로 증가 (탐험 강화)

# Value Loss 및 Gradient Clipping
VALUE_LOSS_COEF = 0.5  # Value loss coefficient (안정성 유지)
MAX_GRAD_NORM = 0.5  # Gradient clipping (폭발 방지)

# 경로 설정
LOG_DIR = "./logs"
MODEL_DIR = "./models"
