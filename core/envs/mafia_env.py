import functools
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from typing import Dict, Any, List, Optional

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from core.engine.game import MafiaGame
from core.agents.base_agent import BaseAgent
from core.agents.rule_base_agent import RuleBaseAgent
from core.agents.llm_agent import LLMAgent
from config import config, Role, Phase, EventType, ActionType
from core.engine.state import GameStatus, GameAction, PlayerStatus, GameEvent
from core.envs.encoders import MDPEncoder, POMDPEncoder, BaseEncoder


class EnvAgent(BaseAgent):
    """
    Environment internal agent placeholder to satisfy MafiaGame requirements.
    This agent does not perform any logic; it just holds state.
    """

    def get_action(self, status: GameStatus) -> GameAction:
        return GameAction(target_id=-1, claim_role=None)


class MafiaEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "mafia_v1"}

    def __init__(self, render_mode=None, encoder_map: Dict[int, BaseEncoder] = None):
        self.possible_agents = [f"player_{i}" for i in range(config.game.PLAYER_COUNT)]
        self.agents = self.possible_agents[:]
        self.render_mode = render_mode

        # Create dummy agents for the engine
        # MafiaGame expects a list of BaseAgent instances
        self.game_agents = [EnvAgent(i) for i in range(config.game.PLAYER_COUNT)]
        self.game = MafiaGame(agents=self.game_agents)

        # Internal agents for environment internalization (RBA/LLM)
        self.internal_agents: Dict[int, Any] = {}

        # === [Encoder Setup] ===
        # 직접 생성하지 않고 외부에서 주입받은 encoder_map 사용
        self.encoder_map = encoder_map or {i: MDPEncoder() for i in range(config.game.PLAYER_COUNT)}

        # === [Multi-Discrete Action Space] ===
        # 형태: [Target, Role]
        # - Target: 0=None, 1~8=Player 0~7 (9개)
        # - Role: 0=None, 1~4=Role Enum (5개)
        self.action_spaces = {
            agent: spaces.MultiDiscrete(config.game.ACTION_DIMS)
            for agent in self.possible_agents
        }

        # Observation Space: Dynamic based on Encoder
        self.observation_spaces = {}
        for agent in self.possible_agents:
            pid = self._agent_to_id(agent)
            obs_dim = self.encoder_map[pid].observation_dim
            
            self.observation_spaces[agent] = spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=-1, high=1, shape=(obs_dim,), dtype=np.float32
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(14,), dtype=np.int8
                    ),
                }
            )

        # 상태 추적 변수 (보상 계산용)
        self.last_executed_player = None
        self.last_killed_player = None
        self.last_investigated_player = None
        self.last_protected_player = None
        self.attack_was_blocked = False

    def reset(self, seed=None, options=None):
        """
        Resets the environment.
        """
        self.agents = self.possible_agents[:]
        self.game.reset()

        # [REMOVED] event dumping logic for speed optimization
        self.last_history_idx = len(self.game.history)

        # 상태 초기화
        self.last_executed_player = None
        self.last_killed_player = None
        self.last_investigated_player = None
        self.last_protected_player = None
        self.attack_was_blocked = False

        # 내부 에이전트(RBA/LLM) 초기화 및 동기화
        self.internal_agents.clear()
        for player in self.game.players:
            # 기본적으로 RuleBaseAgent 사용
            # 추후 config 등을 통해 LLM 사용 여부 결정 가능
            self.internal_agents[player.id] = RuleBaseAgent(player.id, player.role)

        observations = {}
        infos = {}

        self.last_history_idx = 0
        new_events = [
            e.model_dump() for e in self.game.history[self.last_history_idx :]
        ]
        self.last_history_idx = len(self.game.history)

        for agent in self.agents:
            pid = self._agent_to_id(agent)
            observations[agent] = {
                "observation": self._encode_observation(pid),
                "action_mask": self._get_action_mask(pid),  # 액션 마스크 포함 필수
            }
            infos[agent] = {
                "log_events": new_events,
                "role": self.game.players[pid].role,
            }

        return observations, infos

    def step(self, actions):
        # 환경 내재화: 액션이 없는 에이전트(RBA/LLM)의 액션 생성
        # 현재 생존한 모든 플레이어에 대해 확인
        current_alive_ids = [p.id for p in self.game.players if p.alive]

        for pid in current_alive_ids:
            agent_name = self._id_to_agent(pid)

            # 외부에서 액션이 주어지지 않았거나, 무의미한 액션([0,0])인 경우 내부 로직 실행
            should_act = False

            if pid in self.internal_agents:
                if agent_name not in actions:
                    should_act = True
                else:
                    act = np.array(actions[agent_name])
                    if np.all(act == -1):
                        should_act = True

            if should_act:
                status = self.game.get_game_status(pid)
                internal_agent = self.internal_agents[pid]

                # 에이전트 로직 수행 (Role은 Reset 시 이미 동기화됨)
                game_action = internal_agent.get_action(status)

                # 결과를 Multi-Discrete 벡터로 변환하여 actions에 추가
                actions[agent_name] = game_action.to_multi_discrete()

        # Convert string keys to int keys for the engine
        engine_actions = {}
        for agent_id, action in actions.items():
            pid = self._agent_to_id(agent_id)
            if isinstance(action, (list, np.ndarray)):
                engine_actions[pid] = GameAction.from_multi_discrete(action)
            elif isinstance(action, GameAction):
                engine_actions[pid] = action
            elif isinstance(action, dict):
                engine_actions[pid] = action
            else:
                # Fallback or error
                pass

        # 턴 진행 전 상태 저장 (보상 계산용)
        prev_phase = self.game.phase
        prev_alive_count = sum(1 for p in self.game.players if p.alive)

        # 게임 진행
        status, is_over, is_win = self.game.step_phase(engine_actions)

        # [REMOVED] event dumping logic for speed optimization
        new_events = [
            e.model_dump() for e in self.game.history[self.last_history_idx :]
        ]
        self.last_history_idx = len(self.game.history)

        # 상태 변화 추적
        self._track_state_changes(prev_phase, prev_alive_count, engine_actions)

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        episode_metrics = {}
        if is_over:
            episode_metrics = self._calculate_episode_metrics()

        for agent in self.agents:
            pid = self._agent_to_id(agent)

            observations[agent] = {
                "observation": self._encode_observation(pid),
                "action_mask": self._get_action_mask(pid),
            }

            if is_over:
                _, my_win = self.game.check_game_over(player_id=pid)
            else:
                my_win = False

            rewards[agent] = self._calculate_reward(
                pid, prev_phase, engine_actions.get(pid), is_over, my_win
            )
            terminations[agent] = is_over
            truncations[agent] = False

            agent_info = {
                "day": status.day,
                "phase": status.phase,
                "role": self.game.players[pid].role,
                "win": my_win,
                "log_events": new_events,
            }

            if is_over:
                agent_info["episode_metrics"] = episode_metrics

            infos[agent] = agent_info

        if is_over:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _calculate_episode_metrics(self) -> Dict[str, float]:
        """게임 종료 시 통계 지표 계산"""
        metrics = {}
        game = self.game

        # 1. Game Duration & Winner
        metrics["Game/Duration"] = game.day

        mafia_won = False
        citizen_won = False

        last_event = game.history[-1] if game.history else None
        if last_event and last_event.phase == Phase.GAME_END:
            citizen_won_game = last_event.value
            if citizen_won_game:
                citizen_won = True
            else:
                mafia_won = True

        metrics["Game/Mafia_Win"] = 1.0 if mafia_won else 0.0
        metrics["Game/Citizen_Win"] = 1.0 if citizen_won else 0.0

        # Citizen Survival Rate
        initial_citizens = sum(1 for p in game.players if p.role != Role.MAFIA)
        current_citizens = sum(
            1 for p in game.players if p.role != Role.MAFIA and p.alive
        )
        metrics["Game/Citizen_Survival_Rate"] = (
            current_citizens / initial_citizens if initial_citizens > 0 else 0.0
        )

        # Action Stats
        # Action Counter
        mafia_kill_attempts = 0
        mafia_kill_success = 0
        doctor_save_success = 0
        doctor_self_heal = 0
        doctor_total_protects = 0

        police_investigations = 0
        police_finds = 0

        # Vote Counter
        vote_total = 0
        vote_abstain = 0
        mafia_betrayal = 0
        citizen_correct_vote = 0

        mafia_votes = 0
        citizen_votes = 0

        # Execution Counter
        execution_total = 0
        mafia_executed = 0
        citizen_sacrificed = 0

        # Night Interactions
        night_events = [e for e in game.history if e.phase == Phase.NIGHT]

        for d in range(1, game.day + 1):
            day_night_events = [e for e in night_events if e.day == d]
            kill_event = next(
                (e for e in day_night_events if e.event_type == EventType.KILL), None
            )
            protect_event = next(
                (e for e in day_night_events if e.event_type == EventType.PROTECT), None
            )

            if protect_event:
                doctor_total_protects += 1
                if protect_event.actor_id == protect_event.target_id:
                    doctor_self_heal += 1

            if kill_event:
                mafia_kill_attempts += 1
                is_saved = False

                if protect_event and kill_event.target_id == protect_event.target_id:
                    is_saved = True
                    doctor_save_success += 1

                if not is_saved:
                    mafia_kill_success += 1

        # Event Loop
        for event in game.history:
            if event.event_type == EventType.POLICE_RESULT:
                police_investigations += 1
                if event.value == Role.MAFIA:
                    police_finds += 1

            elif event.event_type == EventType.VOTE:
                vote_total += 1
                actor = game.players[event.actor_id]

                if actor.role == Role.MAFIA:
                    mafia_votes += 1
                else:
                    citizen_votes += 1

                if event.target_id == -1:
                    vote_abstain += 1
                else:
                    target = game.players[event.target_id]
                    if actor.role == Role.MAFIA and target.role == Role.MAFIA:
                        mafia_betrayal += 1

                    if actor.role != Role.MAFIA and target.role == Role.MAFIA:
                        citizen_correct_vote += 1

            elif event.event_type == EventType.EXECUTE:
                if event.target_id != -1:
                    execution_total += 1
                    target = game.players[event.target_id]
                    if target.role == Role.MAFIA:
                        mafia_executed += 1
                    else:
                        citizen_sacrificed += 1

        metrics["Vote/Abstain_Rate"] = (
            vote_abstain / vote_total if vote_total > 0 else 0.0
        )
        metrics["Vote/Mafia_Betrayal_Rate"] = (
            mafia_betrayal / mafia_votes if mafia_votes > 0 else 0.0
        )
        metrics["Vote/Citizen_Accuracy_Rate"] = (
            citizen_correct_vote / citizen_votes if citizen_votes > 0 else 0.0
        )

        metrics["Action/Doctor_Save_Rate"] = (
            doctor_save_success / mafia_kill_attempts
            if mafia_kill_attempts > 0
            else 0.0
        )
        metrics["Action/Doctor_Self_Heal_Rate"] = (
            doctor_self_heal / doctor_total_protects
            if doctor_total_protects > 0
            else 0.0
        )

        metrics["Action/Police_Find_Rate"] = (
            police_finds / police_investigations if police_investigations > 0 else 0.0
        )
        metrics["Action/Mafia_Kill_Success_Rate"] = (
            mafia_kill_success / mafia_kill_attempts if mafia_kill_attempts > 0 else 0.0
        )

        metrics["Game/Execution_Frequency"] = (
            execution_total / game.day if game.day > 0 else 0.0
        )

        metrics["Vote/Mafia_Lynch_Rate"] = (
            mafia_executed / execution_total if execution_total > 0 else 0.0
        )
        metrics["Vote/Citizen_Sacrifice_Rate"] = (
            citizen_sacrificed / execution_total if execution_total > 0 else 0.0
        )

        return metrics

    def render(self):
        """게임 상태 렌더링"""
        phase_names = {
            Phase.DAY_DISCUSSION: "Discussion",
            Phase.DAY_VOTE: "Vote",
            Phase.DAY_EXECUTE: "Execute",
            Phase.NIGHT: "Night",
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

    def get_game_status(self, agent_id):
        """
        특정 에이전트 시점의 게임 상태를 반환합니다.
        runner.py 등 외부에서 에이전트의 행동 결정을 위해 호출합니다.
        """
        return self.game.get_game_status(agent_id)

    def _agent_to_id(self, agent_str):
        return int(agent_str.split("_")[1])

    def _id_to_agent(self, agent_id):
        return f"player_{agent_id}"

    # === 상태 추적 ===

    def _track_state_changes(self, prev_phase, prev_alive_count, actions):
        """
        이전 페이즈와 현재 상태를 비교하여 중요한 이벤트를 추적
        """
        current_alive_count = sum(1 for p in self.game.players if p.alive)

        # 처형 추적
        if prev_phase == Phase.DAY_VOTE and self.game.phase == Phase.DAY_EXECUTE:
            # 투표로 처형된 플레이어 찾기
            for player in self.game.players:
                if not player.alive and player.vote_count > 0:
                    self.last_executed_player = player
                    break

        # 밤 사망 추적
        if prev_phase == Phase.NIGHT and current_alive_count < prev_alive_count:
            # 마피아에게 죽은 플레이어 찾기
            for player in self.game.players:
                if not player.alive and player != self.last_executed_player:
                    self.last_killed_player = player
                    break

        # 보호 성공 추적 (밤인데 아무도 안 죽음)
        if prev_phase == Phase.NIGHT and current_alive_count == prev_alive_count:
            mafia_target = None
            doctor_target = None

            for pid, action in actions.items():
                player = self.game.players[pid]
                if not player.alive:
                    continue

                target_id = -1
                if isinstance(action, dict):
                    target_id = action.get("target_id", -1)
                elif hasattr(action, "target_id"):
                    target_id = action.target_id

                if target_id != -1 and 0 <= target_id < len(self.game.players):
                    if player.role == Role.MAFIA:
                        mafia_target = target_id
                    elif player.role == Role.DOCTOR:
                        doctor_target = target_id

            # 마피아 타겟과 의사 타겟이 같으면 보호 성공
            if mafia_target is not None and doctor_target is not None:
                if mafia_target == doctor_target:
                    self.attack_was_blocked = True
                    self.last_protected_player = self.game.players[doctor_target]
        else:
            self.attack_was_blocked = False
            self.last_protected_player = None

    # === 단순화된 보상 시스템 ===

    def _calculate_reward(self, agent_id, prev_phase, my_action, done, win):
        """
        3가지 핵심 보상만 사용:
        1. 게임 승패 (주요 신호)
        2. 마피아 제거 기여
        3. 역할별 핵심 목표 달성
        """
        reward = 0.0
        agent = self.game.players[agent_id]

        # === 보상 1: 게임 승패 (가장 중요) ===
        if done:
            reward += 10.0 if win else -10.0

        # === 보상 2: 마피아 제거 기여 ===
        if prev_phase == Phase.DAY_VOTE and self.last_executed_player:
            reward += self._calculate_execution_reward(agent_id, my_action)

        # === 보상 3: 역할별 핵심 목표 달성 ===
        if prev_phase == Phase.NIGHT:
            reward += self._calculate_night_action_reward(agent_id, my_action)

        return reward

    def _calculate_execution_reward(self, agent_id, my_action):
        """
        처형 단계에서 보상 계산
        - 마피아 처형 성공 시 투표한 시민팀에게 보상
        - 시민 처형 시 페널티
        """
        if not self.last_executed_player:
            return 0.0

        reward = 0.0
        agent = self.game.players[agent_id]
        executed = self.last_executed_player

        # 내가 이 플레이어에게 투표했는가?
        did_vote = False
        if my_action:
            target_id = -1
            if isinstance(my_action, dict):
                target_id = my_action.get("target_id", -1)
            elif hasattr(my_action, "target_id"):
                target_id = my_action.target_id

            did_vote = target_id == executed.id

        if not did_vote:
            return 0.0

        # 역할에 따른 보상
        if agent.role != Role.MAFIA:
            # 시민팀: 마피아 처형 시 보상
            if executed.role == Role.MAFIA:
                reward += 4.0
            # 중요 역할 처형 시 큰 페널티
            elif executed.role in [Role.POLICE, Role.DOCTOR]:
                reward -= 3.0
            # 일반 시민 처형 시 작은 페널티
            elif executed.role == Role.CITIZEN:
                reward -= 1.0
        else:
            # 마피아: 시민 처형 시 보상, 동료 마피아 처형 시 페널티
            if executed.role == Role.MAFIA:
                reward -= 5.0  # 동료 배신
            elif executed.role in [Role.POLICE, Role.DOCTOR]:
                reward += 2.0  # 중요 역할 제거
            else:
                reward += 1.0  # 일반 시민 제거

        return reward

    def _calculate_night_action_reward(self, agent_id, my_action):
        """
        밤 행동 보상 계산
        - 마피아: 성공적인 킬
        - 경찰: 마피아 발견
        - 의사: 보호 성공
        """
        if not my_action:
            return 0.0

        reward = 0.0
        agent = self.game.players[agent_id]

        target_id = -1
        if isinstance(my_action, dict):
            target_id = my_action.get("target_id", -1)
        elif hasattr(my_action, "target_id"):
            target_id = my_action.target_id

        if target_id == -1 or target_id >= len(self.game.players):
            return 0.0

        target = self.game.players[target_id]

        # 마피아: 킬 성공
        if agent.role == Role.MAFIA:
            if self.last_killed_player and self.last_killed_player.id == target_id:
                # 중요 역할 제거 시 더 큰 보상
                if target.role == Role.POLICE:
                    reward += 4.0
                elif target.role == Role.DOCTOR:
                    reward += 3.0
                else:
                    reward += 2.0

        # 경찰: 마피아 조사 성공
        elif agent.role == Role.POLICE:
            if target.role == Role.MAFIA and target.alive:
                reward += 3.0

        # 의사: 보호 성공
        elif agent.role == Role.DOCTOR:
            if self.attack_was_blocked and self.last_protected_player:
                if self.last_protected_player.id == target_id:
                    # 중요 역할 보호 시 더 큰 보상
                    if target.role == Role.POLICE:
                        reward += 4.0
                    elif target.role == Role.DOCTOR:
                        reward += 3.0
                    else:
                        reward += 2.5

        return reward

    # === 관측 및 액션 마스크 ===

    def _encode_observation(
        self, agent_id: int, target_event: Optional[GameEvent] = None
    ) -> np.ndarray:
        """
        Delegate observation encoding to the assigned strategy (Encoder).
        """
        return self.encoder_map[agent_id].encode(self.game, agent_id)

    def _get_action_mask(self, agent_id):
        """
        에이전트가 현재 상태에서 수행할 수 있는 유효한 행동 마스크를 생성합니다.
        마스크는 [Target(9), Role(5)] 형태로 총 14차원입니다.

        타겟 마스크 (_target_mask, 9차원):
            - mask[0]: 아무도 지목하지 않음 (PASS)
            - mask[1] ~ mask[8]: Player 0 ~ 7 지목

        역할 마스크 (_role_mask, 5차원):
            - mask[0]: 역할을 주장하지 않음
            - mask[1] ~ mask[4]: 시민, 경찰, 의사, 마피아 역할 주장
        """
        status = self.game.get_game_status(agent_id)
        agent = self.game.players[agent_id]

        _target_mask = np.zeros(9, dtype=np.int8)
        _role_mask = np.zeros(5, dtype=np.int8)

        if not agent.alive:
            return np.concatenate([_target_mask, _role_mask])

        _target_mask[0] = 1
        phase = status.phase
        is_active_night_role = phase == Phase.NIGHT and agent.role in [
            Role.MAFIA,
            Role.POLICE,
            Role.DOCTOR,
        ]

        if is_active_night_role:
            _target_mask[0] = 0
        else:
            _target_mask[0] = 1

        _role_mask[0] = 1

        valid_targets = {p.id for p in self.game.players if p.alive}
        phase = status.phase

        if phase == Phase.NIGHT:
            if agent.role == Role.MAFIA:
                mafia_team_ids = {
                    p.id for p in self.game.players if p.role == Role.MAFIA
                }
                valid_targets -= mafia_team_ids
            elif agent.role == Role.POLICE:
                valid_targets.discard(agent_id)
            elif agent.role == Role.CITIZEN:
                valid_targets.clear()

        elif phase == Phase.DAY_VOTE:
            valid_targets.discard(agent_id)

        elif phase == Phase.DAY_DISCUSSION:
            _role_mask[1:] = 1

        if not valid_targets:
            _target_mask[1:] = 0
        else:
            for target_id in valid_targets:
                _target_mask[target_id + 1] = 1

        return np.concatenate([_target_mask, _role_mask])
