# 역할 지정
ROLE_CITIZEN = 0
ROLE_POLICE = 1
ROLE_DOCTOR = 2
ROLE_MAFIA = 3

# 게임 단계 (PHASE_DAY_CLAIM removed - merged with PHASE_DAY_DISCUSSION)
PHASE_DAY_DISCUSSION = "day_discussion"  # 낮: 토론 및 주장 단계
PHASE_DAY_VOTE = "day_vote"  # 낮: 투표 단계
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

# 학습 설정 (Hyperparameters)
LR = 0.0003  # Learning Rate
GAMMA = 0.99  # Discount Factor
EPS_CLIP = 0.2  # PPO Clip range
K_EPOCHS = 4  # Update epochs
BATCH_SIZE = 32

# 경로 설정
LOG_DIR = "./logs"
MODEL_DIR = "./models"
