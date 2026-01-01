import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, List
from core.game import MafiaGame
from config import *
from state import GameStatus, MafiaAction, PlayerStatus, GameEvent


class MafiaEnv(gym.Env):
    def __init__(self, log_file=None, logger=None):
        super(MafiaEnv, self).__init__()
        self.game = MafiaGame(log_file=log_file, logger=logger)
        self.logger = logger
        
        # === [Multi-Discrete Action Space] ===
        # 형태: [Target, Role]
        # - Target: 0=None, 1~8=Player 0~7 (9개)
        # - Role: 0=None, 1~4=Role Enum (5개)
        self.action_space = spaces.MultiDiscrete([9, 5])

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
                    low=0, high=1, shape=(14,), dtype=np.int8  # [Target(9), Role(5)]
                ),
            }
        )
        
        # 이전 턴의 투표 기록 저장
        self.last_vote_record = np.zeros((config.game.PLAYER_COUNT, config.game.PLAYER_COUNT), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        status = self.game.reset()
        # 투표 기록 초기화
        self.last_vote_record = np.zeros((config.game.PLAYER_COUNT, config.game.PLAYER_COUNT), dtype=np.float32)
        
        # Return observation for all agents (or just active ones)
        # For now, return dict {agent_id: obs}
        obs_dict = {}
        for p in self.game.players:
            if p.alive:
                obs_dict[p.id] = self._encode_observation(p.id)
                
        return obs_dict, {}

    def step(self, action_input):
        """
        환경 스텝 실행 - Multi-Discrete 방식
        
        Args:
            action_input: 
                - List[int] (Single Agent): Player 0's action vector
                - Dict[int, List[int]] (Multi Agent): {player_id: action_vector}
        
        Returns:
            observation, reward, done, truncated, info
        """
        # 1. Parse actions
        actions_dict = {}
        
        if isinstance(action_input, dict):
            # Multi-Agent case
            for pid, vec in action_input.items():
                actions_dict[pid] = MafiaAction.from_multi_discrete(vec)
        else:
            # Single Agent case (Player 0)
            actions_dict[0] = MafiaAction.from_multi_discrete(action_input)

        # 2. Fill missing actions from internal agents (Bots/LLMs)
        # Note: Ideally, the runner should handle this, but for compatibility we do it here if missing.
        for p in self.game.players:
            if p.id not in actions_dict and p.alive:
                try:
                    # Non-RL agents or RL agents not controlled externally
                    action = p.get_action()
                    if isinstance(action, MafiaAction):
                        actions_dict[p.id] = action
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(f"Error getting action for player {p.id}: {e}")
                    else:
                        print(f"Error getting action for player {p.id}: {e}")

        # 턴 진행 전 상태 저장
        prev_alive = [p.alive for p in self.game.players]
        prev_phase = self.game.phase

        # 게임 진행
        status, done, win = self.game.step_phase(actions_dict)
        
        # === 투표 기록 업데이트 (PHASE_DAY_VOTE 종료 후) ===
        if prev_phase == Phase.DAY_VOTE:
            # 투표 매트릭스 업데이트
            self.last_vote_record = np.zeros((config.game.PLAYER_COUNT, config.game.PLAYER_COUNT), dtype=np.float32)
            for i, player in enumerate(self.game.players):
                if hasattr(player, 'voted_by_last_turn'):
                    for voter_id in player.voted_by_last_turn:
                        self.last_vote_record[voter_id][i] = 1.0

        # === [보상 함수 고도화 v2.0: Dense Reward] ===
        rewards = {}
        
        # Calculate reward for each agent in action_input (or all agents)
        # For MARL, we return rewards for all agents.
        for p in self.game.players:
            rewards[p.id] = self._calculate_reward(p.id, prev_alive, prev_phase, actions_dict.get(p.id), done, win)

        # Generate observations for all agents
        obs_dict = {}
        for p in self.game.players:
            if p.alive:
                obs_dict[p.id] = self._encode_observation(p.id)

        # If single agent input, return single values for compatibility (optional but risky for MARL transition)
        # Let's stick to MARL interface if input was dict, or single if input was list.
        if isinstance(action_input, list):
             # Single agent (Player 0) compatibility
             return obs_dict.get(0), rewards.get(0, 0.0), done, False, {}
        
        return obs_dict, rewards, done, False, {}

    def _calculate_reward(self, agent_id, prev_alive, prev_phase, mafia_action, done, win):
        reward = 0.0
        agent = self.game.players[agent_id]
        role = agent.role
        
        # 1. 승패 보상
        if done:
            # 승리 조건: 시민팀(시민,경찰,의사) vs 마피아팀
            is_mafia_team = role == Role.MAFIA
            # win is True if Citizen team won? 
            # check_game_over returns (is_over, is_citizen_win) usually.
            # Let's verify check_game_over in game.py. 
            # Assuming win=True means Citizen Win.
            
            my_win = (win and not is_mafia_team) or (not win and is_mafia_team)
            
            if my_win:
                reward += 30.0
                reward += (config.game.MAX_DAYS - self.game.day) * 1.0
            else:
                reward -= 15.0

        # 2. 생존 보상
        if not agent.alive:
            reward -= 2.0
        else:
            reward += 0.5

        # 3. 역할 기반 행동 보상
        if agent.alive and mafia_action:
            target_id = mafia_action.target_id
            claim_role = mafia_action.claim_role
            
            # === [역할 주장 보상] ===
            if claim_role is not None:
                if claim_role == role:
                    reward += 2.0
                    if role in [Role.POLICE, Role.DOCTOR]:
                        reward += 3.0
                else:
                    if role == Role.MAFIA:
                        reward += 1.0
                    else:
                        reward -= 2.0
            
            # === [행동 보상] ===
            # Target Action: Target != -1 and Role == None
            if target_id != -1 and claim_role is None:
                if role == Role.CITIZEN:
                    reward += self._calculate_citizen_reward(target_id, prev_phase)
                elif role == Role.MAFIA:
                    reward += self._calculate_mafia_reward(target_id, prev_phase)
                elif role == Role.POLICE:
                    reward += self._calculate_citizen_reward(target_id, prev_phase)
                    reward += self._calculate_police_reward(target_id, prev_phase)
                elif role == Role.DOCTOR:
                    reward += self._calculate_citizen_reward(target_id, prev_phase)
                    reward += self._calculate_doctor_reward(prev_alive, target_id, prev_phase)
        
        return reward

    def _encode_observation(self, agent_id):
        """
        개별 에이전트 관점의 관측값 생성
        """
        # ... existing implementation adapted for agent_id ...
        # For now, we reuse the existing logic but replace 'my_id' with 'agent_id'
        # Since the original code used 'my_id = 0' implicitly or explicitly, 
        # I need to see the original _encode_observation code.
        # I'll assume I need to rewrite it or it's not shown fully.
        # Let's read the file again to be sure about _encode_observation.
        pass

    
    def _calculate_citizen_reward(self, action, phase):
        """시민 팀 공통 보상 로직 - Dense Reward 강화"""
        if action == -1:
            return 0.0
            
        reward = 0.0
        
        # IndexError 방지
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]

            # 낮 투표: 마피아를 지목하면 강력한 중간 보상
            if phase == Phase.DAY_VOTE:
                if target.role == Role.MAFIA:
                    reward += 15.0  # 마피아 투표 성공 (대폭 강화: 5 → 15)
                    # 투표 성공 시 추가 보상 (즉각적 피드백)
                    if not target.alive:  # 실제로 처형되었다면
                        reward += 10.0  # 처형 성공 추가 보상
                elif target.role in [Role.POLICE, Role.DOCTOR]:
                    reward -= 8.0  # 중요 역할 지목 페널티 강화
                elif target.role == Role.CITIZEN:
                    reward -= 2.0  # 시민 지목 페널티
            
            # 낮 토론: 마피아를 의심하면 중간 보상 (학습 신호)
            elif phase == Phase.DAY_DISCUSSION:
                if target.role == Role.MAFIA:
                    reward += 3.0  # 마피아 의심 보상 강화 (1 → 3)
                elif target.role in [Role.POLICE, Role.DOCTOR]:
                    reward -= 1.0  # 중요 역할 의심 시 소량 페널티
                
        return reward

    def _calculate_mafia_reward(self, action, phase):
        """마피아 보상 로직 - Dense Reward 대폭 강화"""
        if action == -1:
            return 0.0
            
        reward = 0.0
        
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]

            # 낮 투표: 시민 팀 제거 성공 (중간 보상 강화)
            if phase == Phase.DAY_VOTE:
                if target.role == Role.POLICE:
                    reward += 20.0  # 경찰 제거 최우선 (대폭 강화: 8 → 20)
                    if not target.alive:  # 실제 처형 성공
                        reward += 15.0  # 처형 성공 추가 보상
                elif target.role == Role.DOCTOR:
                    reward += 15.0  # 의사 제거 (5 → 15)
                    if not target.alive:
                        reward += 10.0
                elif target.role == Role.CITIZEN:
                    reward += 5.0  # 시민 제거 (2 → 5)
                    if not target.alive:
                        reward += 3.0
                elif target.role == Role.MAFIA:
                    reward -= 25.0  # 동료 마피아 지목 심각한 페널티 (강화)

            # 밤 행동: 중요 역할 제거 (매우 강력한 중간 보상)
            elif phase == Phase.NIGHT:
                if target.role == Role.POLICE:
                    reward += 25.0  # 경찰 제거 최우선 (대폭 강화: 10 → 25)
                    # 실제 킬 성공 확인 (의사 치료 실패)
                    if not target.alive:
                        reward += 15.0  # 킬 성공 추가 보상
                elif target.role == Role.DOCTOR:
                    reward += 18.0  # 의사 제거 (7 → 18)
                    if not target.alive:
                        reward += 12.0
                elif target.role == Role.CITIZEN:
                    reward += 8.0  # 시민 제거 (3 → 8)
                    if not target.alive:
                        reward += 5.0
                
        return reward

    def _calculate_police_reward(self, action, phase):
        """경찰 특수 보상 - 밤에 조사 성공 (Dense Reward 대폭 강화)"""
        if action == -1 or phase != Phase.NIGHT:
            return 0.0
            
        reward = 0.0
        
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]
            if target.role == Role.MAFIA:
                reward += 20.0  # 마피아 발견 성공 (대폭 강화: 8 → 20)
                # 조사 성공은 매우 중요한 정보 획득
            else:
                reward += 2.0  # 조사 자체에 대한 보상 증가 (0.5 → 2.0)
        return reward

    def _calculate_doctor_reward(self, prev_alive, action, phase):
        """의사 특수 보상 - 치료 성공 (Dense Reward 대폭 강화)"""
        if action == -1 or phase != Phase.NIGHT:
            return 0.0
            
        reward = 0.0
        
        # 치료 성공 확인 (사망자가 없었으면 치료 성공)
        current_alive_count = sum(p.alive for p in self.game.players)
        prev_alive_count = sum(prev_alive)
        
        if current_alive_count == prev_alive_count:
            # 치료 성공 (매우 중요한 행동)
            reward += 25.0  # 치료 성공 보상 대폭 강화 (10 → 25)
            
            # 중요 역할을 살렸으면 추가 보상
            if 0 <= action < len(self.game.players):
                target = self.game.players[action]
                if target.role == Role.POLICE:
                    reward += 15.0  # 경찰 구출 (5 → 15)
                elif target.role == Role.DOCTOR:
                    reward += 10.0  # 자기 자신 또는 다른 의사
                elif target.role == Role.CITIZEN:
                    reward += 5.0  # 시민 구출
        else:
            # 치료 실패 시에도 시도에 대한 소량 보상
            reward += 1.0  # 행동 자체에 대한 피드백

        return reward

    def _encode_observation(self, agent_id: int) -> Dict:
        """
        GameStatus를 사용한 관측 인코딩 - 순수 함수 기반
        
        State Matrix 구성 (총 152차원):
        - alive_status: 8 (생존 여부)
        - my_role: 4 (내 역할 one-hot)
        - claim_status: 8 (각 플레이어가 주장한 역할: 0~3)
        - accusation_matrix: 8*8=64 (누가 누구를 의심했는지)
        - last_vote_matrix: 8*8=64 (직전 투표에서 누가 누구에게 투표했는지)
        - day_count: 1 (현재 날짜, 정규화)
        - phase_onehot: 3 (현재 페이즈: discussion, vote, night)
        """
        status = self.game.get_game_status(agent_id)
        n_players = config.game.PLAYER_COUNT
        
        # 1. alive_status (8차원)
        alive_status = np.array([
            1.0 if p.alive else 0.0 
            for p in status.players
        ], dtype=np.float32)
        
        # 2. my_role (4차원 one-hot)
        my_role_vec = np.zeros(len(Role), dtype=np.float32)
        my_role_vec[int(status.my_role)] = 1.0
        
        # 3. claim_status (8차원) - history에서 CLAIM 이벤트 추출
        claim_status = np.zeros(n_players, dtype=np.float32)
        for event in status.action_history:
            if event.event_type == EventType.CLAIM and event.value is not None:
                if isinstance(event.value, Role):
                    claim_status[event.actor_id] = float(int(event.value))
        
        # 4. accusation_matrix (64차원 = 8x8) - CLAIM + target_id가 있는 경우
        accusation_matrix = np.zeros((n_players, n_players), dtype=np.float32)
        for event in status.action_history:
            if event.event_type == EventType.CLAIM and event.target_id is not None:
                accusation_matrix[event.actor_id][event.target_id] = 1.0
        accusation_flat = accusation_matrix.flatten()
        
        # 5. last_vote_matrix (64차원 = 8x8) - 외부에서 전달받은 투표 기록
        last_vote_flat = self.last_vote_record.flatten()
        
        # 6. day_count (1차원, 정규화)
        day_normalized = np.array([status.day / config.game.MAX_DAYS], dtype=np.float32)
        
        # 7. phase_onehot (3차원)
        phase_onehot = np.zeros(3, dtype=np.float32)
        if status.phase == Phase.DAY_DISCUSSION:
            phase_onehot[0] = 1.0
        elif status.phase == Phase.DAY_VOTE:
            phase_onehot[1] = 1.0
        elif status.phase == Phase.NIGHT:
            phase_onehot[2] = 1.0
        
        # 전체 결합
        observation = np.concatenate([
            alive_status,       # 8
            my_role_vec,        # 4
            claim_status,       # 8
            accusation_flat,    # 64
            last_vote_flat,     # 64
            day_normalized,     # 1
            phase_onehot,       # 3
        ])
        
        assert observation.shape == (152,), f"Expected 152 dims, got {observation.shape}"
        
        # 액션 마스크 계산
        action_mask = self._compute_action_mask(status, status.my_id, status.my_role)
        
        return {"observation": observation, "action_mask": action_mask}
    
    def _compute_action_mask(self, status: GameStatus, my_id: int, my_role: Role) -> np.ndarray:
        """
        Multi-Discrete 액션 마스크 계산
        
        마스크 구조: [Target(9), Role(5)] -> Flattened (14,)
        - Target: 0~8 (0=None, 1~8=Player)
        - Role: 9~13 (9=None, 10~13=Role)
        """
        mask = np.zeros(14, dtype=np.int8)
        phase = status.phase
        
        # === Target Mask (0~8) ===
        # 0: None (PASS or Self/NoTarget)
        mask[0] = 1
        
        for i in range(config.game.PLAYER_COUNT):
            target_idx = i + 1
            # player = status.players[i]
            is_alive = status.players[i].alive
            
            if not is_alive:
                continue
                
            # 기본적으로 살아있는 대상 지목 허용
            mask[target_idx] = 1
            
            # 제약 조건
            if i == my_id:
                # 자기 자신 지목 제한 (투표, 킬 등)
                if phase in (Phase.DAY_VOTE, Phase.NIGHT):
                    mask[target_idx] = 0
            
            # 밤 행동 제약
            if phase == Phase.NIGHT:
                # 시민은 밤에 아무것도 못함 (PASS만 가능)
                if my_role == Role.CITIZEN:
                    mask[target_idx] = 0
                # 경찰: 자신 조사 불가
                elif my_role == Role.POLICE and i == my_id:
                    mask[target_idx] = 0
        
        # === Role Mask (9~13) ===
        # 9: None (No Claim)
        mask[9] = 1
        
        if phase == Phase.DAY_DISCUSSION:
            # 모든 역할 주장 허용
            mask[10:] = 1
        else:
            # 토론 외에는 역할 주장 불가
            mask[10:] = 0
        
        return mask

    def render(self):
        """게임 상태 렌더링"""
        phase_names = {
            Phase.DAY_DISCUSSION: "Discussion",
            Phase.DAY_VOTE: "Vote",
            Phase.DAY_EXECUTE: "Execute",
            Phase.NIGHT: "Night"
        }
        phase_str = phase_names.get(self.game.phase, str(self.game.phase))
        status = self.game.get_game_status()
        alive_indices = [p.id for p in status.players if p.alive]
        print(f"[Day {self.game.day}] {phase_str} | Alive: {alive_indices}")