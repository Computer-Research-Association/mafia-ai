import functools
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, List, Optional

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from core.game import MafiaGame
from core.agent.baseAgent import BaseAgent
from config import config, Role, Phase, EventType, ActionType
from state import GameStatus, MafiaAction, PlayerStatus, GameEvent

class EnvAgent(BaseAgent):
    """
    Environment internal agent placeholder to satisfy MafiaGame requirements.
    This agent does not perform any logic; it just holds state.
    """
    def update_belief(self, history: List[GameEvent]):
        pass

    def get_action(self) -> MafiaAction:
        return MafiaAction(target_id=-1, claim_role=None)

class MafiaEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "mafia_v1"}

    def __init__(self, render_mode=None, logger=None):
        self.possible_agents = [f"player_{i}" for i in range(config.game.PLAYER_COUNT)]
        self.agents = self.possible_agents[:]
        self.render_mode = render_mode
        self.logger = logger
        
        # Create dummy agents for the engine
        # MafiaGame expects a list of BaseAgent instances
        self.internal_agents = [EnvAgent(i) for i in range(config.game.PLAYER_COUNT)]
        self.game = MafiaGame(agents=self.internal_agents, logger=logger)
        
        # === [Multi-Discrete Action Space] ===
        # 형태: [Target, Role]
        # - Target: 0=None, 1~8=Player 0~7 (9개)
        # - Role: 0=None, 1~4=Role Enum (5개)
        self.action_spaces = {
            agent: spaces.MultiDiscrete([9, 5]) for agent in self.possible_agents
        }
        
        # Observation Space: 공적 정보만 포함 (152차원)
        obs_dim = 152
        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(14,), dtype=np.int8)
            }) for agent in self.possible_agents
        }
        
        # 이전 턴의 투표 기록 저장
        self.last_vote_record = np.zeros((config.game.PLAYER_COUNT, config.game.PLAYER_COUNT), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # PettingZoo API: reset returns (observations, infos)
        self.agents = self.possible_agents[:]
        self.game.reset()
        self.last_vote_record = np.zeros((config.game.PLAYER_COUNT, config.game.PLAYER_COUNT), dtype=np.float32)
        
        observations = {
            agent: self._encode_observation(self._agent_to_id(agent))
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        # Convert string keys to int keys for the engine
        engine_actions = {}
        for agent_id, action in actions.items():
            pid = self._agent_to_id(agent_id)
            if isinstance(action, (list, np.ndarray)):
                engine_actions[pid] = MafiaAction.from_multi_discrete(action)
            elif isinstance(action, MafiaAction):
                engine_actions[pid] = action
            else:
                # Fallback or error
                pass

        # 턴 진행 전 상태 저장 (보상 계산용)
        prev_alive = [p.alive for p in self.game.players]
        prev_phase = self.game.phase

        # 게임 진행
        status, is_over, is_win = self.game.step_phase(engine_actions)
        
        # === 투표 기록 업데이트 (PHASE_DAY_VOTE 종료 후) ===
        if prev_phase == Phase.DAY_VOTE:
            self.last_vote_record = np.zeros((config.game.PLAYER_COUNT, config.game.PLAYER_COUNT), dtype=np.float32)
            for i, player in enumerate(self.game.players):
                if hasattr(player, 'voted_by_last_turn'):
                    for voter_id in player.voted_by_last_turn:
                        self.last_vote_record[voter_id][i] = 1.0

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            pid = self._agent_to_id(agent)
            
            observations[agent] = self._encode_observation(pid)
            rewards[agent] = self._calculate_reward(pid, prev_alive, prev_phase, engine_actions.get(pid), is_over, is_win)
            terminations[agent] = is_over
            truncations[agent] = False
            infos[agent] = {"day": status.day, "phase": status.phase, "win": is_win}

        if is_over:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

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

    def close(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _agent_to_id(self, agent_str):
        return int(agent_str.split("_")[1])
    
    def _id_to_agent(self, agent_id):
        return f"player_{agent_id}"

    # === Helper Methods (Copied and adapted from previous implementation) ===

    def _calculate_reward(self, agent_id, prev_alive, prev_phase, mafia_action, done, win):
        reward = 0.0
        agent = self.game.players[agent_id]
        role = agent.role
        
        # 1. 승패 보상
        if done:
            is_mafia_team = role == Role.MAFIA
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

    def _calculate_citizen_reward(self, action, phase):
        if action == -1: return 0.0
        reward = 0.0
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]
            if phase == Phase.DAY_VOTE:
                if target.role == Role.MAFIA:
                    reward += 15.0
                    if not target.alive: reward += 10.0
                elif target.role in [Role.POLICE, Role.DOCTOR]:
                    reward -= 8.0
                elif target.role == Role.CITIZEN:
                    reward -= 2.0
            elif phase == Phase.DAY_DISCUSSION:
                if target.role == Role.MAFIA:
                    reward += 3.0
                elif target.role in [Role.POLICE, Role.DOCTOR]:
                    reward -= 1.0
        return reward

    def _calculate_mafia_reward(self, action, phase):
        if action == -1: return 0.0
        reward = 0.0
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]
            if phase == Phase.DAY_VOTE:
                if target.role == Role.POLICE:
                    reward += 20.0
                    if not target.alive: reward += 15.0
                elif target.role == Role.DOCTOR:
                    reward += 15.0
                    if not target.alive: reward += 10.0
                elif target.role == Role.CITIZEN:
                    reward += 5.0
                    if not target.alive: reward += 3.0
                elif target.role == Role.MAFIA:
                    reward -= 25.0
            elif phase == Phase.NIGHT:
                if target.role == Role.POLICE:
                    reward += 25.0
                    if not target.alive: reward += 15.0
                elif target.role == Role.DOCTOR:
                    reward += 18.0
                    if not target.alive: reward += 12.0
                elif target.role == Role.CITIZEN:
                    reward += 8.0
                    if not target.alive: reward += 5.0
        return reward

    def _calculate_police_reward(self, action, phase):
        if action == -1 or phase != Phase.NIGHT: return 0.0
        reward = 0.0
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]
            if target.role == Role.MAFIA:
                reward += 20.0
            else:
                reward += 2.0
        return reward

    def _calculate_doctor_reward(self, prev_alive, action, phase):
        if action == -1 or phase != Phase.NIGHT: return 0.0
        reward = 0.0
        current_alive_count = sum(p.alive for p in self.game.players)
        prev_alive_count = sum(prev_alive)
        if current_alive_count == prev_alive_count:
            reward += 25.0
            if 0 <= action < len(self.game.players):
                target = self.game.players[action]
                if target.role == Role.POLICE: reward += 15.0
                elif target.role == Role.DOCTOR: reward += 10.0
                elif target.role == Role.CITIZEN: reward += 5.0
        else:
            reward += 1.0
        return reward

    def _encode_observation(self, agent_id: int) -> Dict:
        status = self.game.get_game_status(agent_id)
        n_players = config.game.PLAYER_COUNT
        
        alive_status = np.array([1.0 if p.alive else 0.0 for p in status.players], dtype=np.float32)
        my_role_vec = np.zeros(len(Role), dtype=np.float32)
        my_role_vec[int(status.my_role)] = 1.0
        
        claim_status = np.zeros(n_players, dtype=np.float32)
        for event in status.action_history:
            if event.event_type == EventType.CLAIM and event.value is not None:
                if isinstance(event.value, Role):
                    claim_status[event.actor_id] = float(int(event.value))
        
        accusation_matrix = np.zeros((n_players, n_players), dtype=np.float32)
        for event in status.action_history:
            if event.event_type == EventType.CLAIM and event.target_id is not None:
                accusation_matrix[event.actor_id][event.target_id] = 1.0
        accusation_flat = accusation_matrix.flatten()
        
        last_vote_flat = self.last_vote_record.flatten()
        day_normalized = np.array([status.day / config.game.MAX_DAYS], dtype=np.float32)
        
        phase_onehot = np.zeros(3, dtype=np.float32)
        if status.phase == Phase.DAY_DISCUSSION: phase_onehot[0] = 1.0
        elif status.phase == Phase.DAY_VOTE: phase_onehot[1] = 1.0
        elif status.phase == Phase.NIGHT: phase_onehot[2] = 1.0
        
        observation = np.concatenate([
            alive_status, my_role_vec, claim_status, accusation_flat, last_vote_flat, day_normalized, phase_onehot
        ])
        
        action_mask = self._compute_action_mask(status, status.my_id, status.my_role)
        return {"observation": observation, "action_mask": action_mask}
    
    def _compute_action_mask(self, status: GameStatus, my_id: int, my_role: Role) -> np.ndarray:
        mask = np.zeros(14, dtype=np.int8)
        phase = status.phase
        
        # Target Mask
        mask[0] = 1
        for i in range(config.game.PLAYER_COUNT):
            target_idx = i + 1
            is_alive = status.players[i].alive
            if not is_alive: continue
            mask[target_idx] = 1
            if i == my_id:
                if phase in (Phase.DAY_VOTE, Phase.NIGHT): mask[target_idx] = 0
            if phase == Phase.NIGHT:
                if my_role == Role.CITIZEN: mask[target_idx] = 0
                elif my_role == Role.POLICE and i == my_id: mask[target_idx] = 0
        
        # Role Mask
        mask[9] = 1
        if phase == Phase.DAY_DISCUSSION: mask[10:] = 1
        else: mask[10:] = 0
        
        return mask