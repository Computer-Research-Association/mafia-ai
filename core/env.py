import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core.game import MafiaGame
import config


class MafiaEnv(gym.Env):
    def __init__(self, log_file=None):
        super(MafiaEnv, self).__init__()
        self.game = MafiaGame(log_file=log_file)
        # Action Space: 0~7번 플레이어 지목 + 8번 기권 (NO_ACTION)
        self.action_space = spaces.Discrete(config.PLAYER_COUNT + 1)

        # Observation Space: 공적 정보만 포함
        # - alive_status: 8 (생존 여부)
        # - my_role: 4 (내 역할 one-hot)
        # - claim_status: 8 (각 플레이어가 주장한 역할: 0~3)
        # - accusation_matrix: 8*8=64 (누가 누구를 의심했는지)
        # - last_vote_matrix: 8*8=64 (직전 투표에서 누가 누구에게 투표했는지)
        # - day_count: 1 (현재 날짜, 정규화)
        # - phase_onehot: 3 (현재 페이즈: discussion, vote, night)
        # Total: 8 + 4 + 8 + 64 + 64 + 1 + 3 = 152
        obs_dim = 152
        
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0,
                    high=1,
                    shape=(obs_dim,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(config.PLAYER_COUNT + 1,), dtype=np.int8  # 0 or 1
                ),
            }
        )
        
        # 이전 턴의 투표 기록 저장
        self.last_vote_record = np.zeros((config.PLAYER_COUNT, config.PLAYER_COUNT), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        status = self.game.reset()
        # 투표 기록 초기화
        self.last_vote_record = np.zeros((config.PLAYER_COUNT, config.PLAYER_COUNT), dtype=np.float32)
        return self._encode_observation(status), {}

    def step(self, action):
        my_id = self.game.players[0].id
        my_role = self.game.players[my_id].role
        
        # 턴 진행 전 상태 저장 (의사 보상 계산용)
        prev_alive = [p.alive for p in self.game.players]
        prev_phase = self.game.phase

        # 기권 액션(config.PLAYER_COUNT)을 -1로 변환
        processed_action = -1 if action == config.PLAYER_COUNT else action

        # 게임 진행
        status, done, win = self.game.process_turn(processed_action)
        
        # === 투표 기록 업데이트 (PHASE_DAY_VOTE 종료 후) ===
        if prev_phase == config.PHASE_DAY_VOTE:
            # 투표 매트릭스 업데이트
            self.last_vote_record = np.zeros((config.PLAYER_COUNT, config.PLAYER_COUNT), dtype=np.float32)
            for i, player in enumerate(self.game.players):
                if hasattr(player, 'voted_by_last_turn'):
                    for voter_id in player.voted_by_last_turn:
                        self.last_vote_record[voter_id][i] = 1.0

        # === [보상 함수 고도화] ===
        reward = 0.0

        # 1. 승패 보상 - 압도적 비중
        if done:
            if win:
                reward += 100.0  # 승리 보상 대폭 증가
                # 빨리 이길수록 추가 보상
                reward += (config.MAX_DAYS - self.game.day_count) * 2.0
            else:
                reward -= 50.0  # 패배 페널티

        # 2. 생존 보상 축소 (의미 없는 보상 최소화)
        if not self.game.players[my_id].alive:
            reward -= 5.0  # 죽음에 대한 명확한 페널티
        else:
            reward += 0.1  # 생존 보상 최소화

        # 3. 역할 기반 행동 보상 (명확한 보상만)
        if self.game.players[my_id].alive and processed_action != -1:
            phase = self.game.phase
            
            if my_role == config.ROLE_CITIZEN:
                reward += self._calculate_citizen_reward(processed_action, phase)
            elif my_role == config.ROLE_MAFIA:
                reward += self._calculate_mafia_reward(processed_action, phase)
            elif my_role == config.ROLE_POLICE:
                reward += self._calculate_citizen_reward(processed_action, phase)
                reward += self._calculate_police_reward(processed_action, phase)
            elif my_role == config.ROLE_DOCTOR:
                reward += self._calculate_citizen_reward(processed_action, phase)
                reward += self._calculate_doctor_reward(prev_alive, processed_action, phase)

        return self._encode_observation(status), reward, done, False, {}

    def _get_action_mask(self):
        mask = np.ones(config.PLAYER_COUNT + 1, dtype=np.int8)
        my_id = self.game.players[0].id
        my_role = self.game.players[my_id].role
        phase = self.game.phase

        for i in range(config.PLAYER_COUNT):
            # 1. 이미 죽은 플레이어는 지목 불가
            if not self.game.players[i].alive:
                mask[i] = 0
                continue

            # 2. 낮 행동 제약 (자신 지목 불가)
            if phase == config.PHASE_DAY_DISCUSSION or phase == config.PHASE_DAY_VOTE:
                if i == my_id:
                    mask[i] = 0

            # 3. 밤 행동 제약
            elif phase == config.PHASE_NIGHT:
                # 마피아: 동료 마피아 지목 불가
                if my_role == config.ROLE_MAFIA:
                    if self.game.players[i].role == config.ROLE_MAFIA:
                        mask[i] = 0
                # 경찰: 자신 조사 불가
                elif my_role == config.ROLE_POLICE:
                    if i == my_id:
                        mask[i] = 0
                # 의사: 자신 치료 가능 (제약 없음)

        # 기권 액션 마스크
        if phase == config.PHASE_DAY_DISCUSSION or phase == config.PHASE_DAY_VOTE:
            mask[config.PLAYER_COUNT] = 1  # 기권 허용
        else:
            mask[config.PLAYER_COUNT] = 0  # 기권 불가

        return mask

    def _calculate_citizen_reward(self, action, phase):
        """시민 팀 공통 보상 로직 - 명확한 보상만"""
        if action == -1:
            return 0.0
            
        reward = 0.0
        
        # IndexError 방지
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]

            # 낮 투표: 마피아를 지목하면 큰 보상
            if phase == config.PHASE_DAY_VOTE:
                if target.role == config.ROLE_MAFIA:
                    reward += 5.0  # 마피아 지목 성공
                elif target.role in [config.ROLE_POLICE, config.ROLE_DOCTOR]:
                    reward -= 3.0  # 중요 역할 지목 페널티
                else:
                    reward -= 0.5  # 시민 지목 페널티
            
            # 낮 토론: 마피아를 의심하면 소량 보상
            elif phase == config.PHASE_DAY_DISCUSSION:
                if target.role == config.ROLE_MAFIA:
                    reward += 1.0
                
        return reward

    def _calculate_mafia_reward(self, action, phase):
        """마피아 보상 로직 - 명확한 보상만"""
        if action == -1:
            return 0.0
            
        reward = 0.0
        
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]

            # 낮 투표: 시민 팀 제거 성공
            if phase == config.PHASE_DAY_VOTE:
                if target.role == config.ROLE_POLICE:
                    reward += 8.0  # 경찰 제거 최우선
                elif target.role == config.ROLE_DOCTOR:
                    reward += 5.0  # 의사 제거
                elif target.role == config.ROLE_CITIZEN:
                    reward += 2.0  # 시민 제거
                elif target.role == config.ROLE_MAFIA:
                    reward -= 10.0  # 동료 마피아 지목 심각한 페널티

            # 밤 행동: 중요 역할 제거
            elif phase == config.PHASE_NIGHT:
                if target.role == config.ROLE_POLICE:
                    reward += 10.0  # 경찰 제거 최우선
                elif target.role == config.ROLE_DOCTOR:
                    reward += 7.0  # 의사 제거
                elif target.role == config.ROLE_CITIZEN:
                    reward += 3.0  # 시민 제거
                
        return reward

    def _calculate_police_reward(self, action, phase):
        """경찰 특수 보상 - 밤에 조사 성공"""
        if action == -1 or phase != config.PHASE_NIGHT:
            return 0.0
            
        reward = 0.0
        
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]
            if target.role == config.ROLE_MAFIA:
                reward += 8.0  # 마피아 발견 성공
            else:
                reward += 0.5  # 조사 자체에 대한 소량 보상
        return reward

    def _calculate_doctor_reward(self, prev_alive, action, phase):
        """의사 특수 보상 - 치료 성공"""
        if action == -1 or phase != config.PHASE_NIGHT:
            return 0.0
            
        reward = 0.0
        
        # 치료 성공 확인 (사망자가 없었으면 치료 성공)
        current_alive_count = sum(self.game.alive_status)
        prev_alive_count = sum(prev_alive)
        
        if current_alive_count == prev_alive_count:
            # 치료 성공
            reward += 10.0
            
            # 경찰을 살렸으면 추가 보상
            if 0 <= action < len(self.game.players):
                target = self.game.players[action]
                if target.role == config.ROLE_POLICE:
                    reward += 5.0

        return reward

    def _encode_observation(self, status):
        """
        공개 정보만을 사용한 관측 인코딩 (RationalCharacter의 belief 사용 금지)
        
        구성:
        1. alive_status (8): 각 플레이어 생존 여부
        2. my_role (4): 내 역할 one-hot (citizen, police, doctor, mafia)
        3. claim_status (8): 각 플레이어가 주장한 역할 (0~3, 정규화)
        4. accusation_matrix (64): 누가 누구를 의심했는지 (8x8 평탄화)
        5. last_vote_matrix (64): 직전 투표 기록 (8x8 평탄화)
        6. day_count (1): 현재 날짜 (정규화)
        7. phase_onehot (3): 현재 페이즈 (discussion, vote, night)
        
        Total: 152차원
        """
        # 1. 생존 상태 (8)
        alive_vector = np.array(status["alive_status"], dtype=np.float32)
        
        # 2. 내 역할 one-hot (4)
        my_role_id = status["roles"][status["id"]]
        role_one_hot = np.zeros(4, dtype=np.float32)
        role_one_hot[my_role_id] = 1.0
        
        # 3. Claim Status (8): 각 플레이어가 주장한 역할
        claim_status = np.zeros(config.PLAYER_COUNT, dtype=np.float32)
        for player in self.game.players:
            # claimed_role 속성이 있으면 사용, 없으면 0 (주장 없음)
            if hasattr(player, 'claimed_role') and player.claimed_role != -1:
                claim_status[player.id] = player.claimed_role / 3.0  # 0~3을 0~1로 정규화
        
        # 4. Accusation Matrix (64): 현재 턴에서 누가 누구를 지목했는지
        accusation_matrix = np.zeros((config.PLAYER_COUNT, config.PLAYER_COUNT), dtype=np.float32)
        for player in self.game.players:
            if hasattr(player, 'claimed_target') and player.claimed_target != -1:
                # player.id가 claimed_target을 지목했음
                if 0 <= player.claimed_target < config.PLAYER_COUNT:
                    accusation_matrix[player.id][player.claimed_target] = 1.0
        accusation_flat = accusation_matrix.flatten()
        
        # 5. Last Vote Matrix (64): 직전 투표 기록 (이미 self.last_vote_record에 저장됨)
        last_vote_flat = self.last_vote_record.flatten()
        
        # 6. Day Count (1): 정규화 (0~1 범위, MAX_DAYS 기준)
        day_normalized = np.array([min(self.game.day_count / config.MAX_DAYS, 1.0)], dtype=np.float32)
        
        # 7. Phase One-hot (3)
        phase_map = {
            config.PHASE_DAY_DISCUSSION: 0,
            config.PHASE_DAY_VOTE: 1,
            config.PHASE_NIGHT: 2,
        }
        phase_idx = phase_map.get(self.game.phase, 0)
        phase_onehot = np.zeros(3, dtype=np.float32)
        phase_onehot[phase_idx] = 1.0
        
        # 전체 observation 결합
        observation = np.concatenate([
            alive_vector,      # 8
            role_one_hot,      # 4
            claim_status,      # 8
            accusation_flat,   # 64
            last_vote_flat,    # 64
            day_normalized,    # 1
            phase_onehot       # 3
        ])
        
        action_mask = self._get_action_mask()
        return {"observation": observation, "action_mask": action_mask}

    def render(self):
        phase_str = (
            ["Claim", "Discussion", "Vote", "Night"][self.game.phase]
            if isinstance(self.game.phase, int)
            else self.game.phase
        )
        alive_indices = [i for i, alive in enumerate(self.game.alive_status) if alive]
        print(f"[Day {self.game.day_count}] {phase_str} | Alive: {alive_indices}")