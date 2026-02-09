import gymnasium as gym
import supersuit as ss
import os
import multiprocessing
import torch

from typing import Dict, Any, List
from pathlib import Path


# [중요] 클래스 밖으로 뺀 함수
# 이 함수는 "로그 파일(logger)" 없이 순수한 게임 환경만 만듭니다.
# 워커 프로세스들은 이 함수를 통해 만들어지므로, 파일 충돌이 안 납니다.
def make_env_for_worker():
    from core.envs.mafia_env import MafiaEnv

    # 워커는 로그를 파일에 안 씀 (logger=None)
    return MafiaEnv()


from core.envs.mafia_env import MafiaEnv
from core.envs.encoders import MDPEncoder, POMDPEncoder, AbsoluteEncoder
from core.agents import AgentBuilder
from core.agents.rl_agent import RLAgent
from core.managers.logger import LogManager
from config import Role, config


class ExperimentManager:
    def __init__(self, args):
        self.args = args
        self.player_configs = getattr(args, "player_configs", [])
        self.mode = args.mode

        # [수정] CLI에서 받은 람다 값을 환경 변수에 주입
        if hasattr(args, "lambda_val") and args.lambda_val is not None:
            os.environ["MAFIA_LAMBDA"] = str(args.lambda_val)
            print(f"[System] REWARD_LAMBDA set to {args.lambda_val} via CLI.")

        self.logger = self._setup_logger()

    def _setup_logger(self) -> LogManager:
        experiment_name = f"llm_{self.mode}"
        if self.player_configs:
            for p_config in self.player_configs:
                if p_config["type"] == "rl":
                    experiment_name = (
                        f"{p_config['algo']}_{p_config['backbone']}_{self.mode}"
                    )
                    break

        # [수정] CLI 인자 `log_dir` 우선 적용
        if hasattr(self.args, "log_dir") and self.args.log_dir:
            log_dir = self.args.log_dir
        else:
            log_dir = str(getattr(self.args, "paths", {}).get("log_dir", "logs"))

        log_events = self.mode != "train"

        # 메인 프로세스는 정상적으로 파일을 씀
        return LogManager(
            experiment_name=experiment_name,
            log_dir=log_dir,
            write_mode=True,
            log_events=log_events,
        )

    # [추가] 환경 내부의 에이전트(Body)에게 고정 역할을 붙여주는 헬퍼 함수
    def _inject_fixed_roles(self, env):
        if not self.player_configs:
            return

        for i, p_config in enumerate(self.player_configs):
            # 1. 설정에서 역할 확인
            role_str = p_config.get("role", "random").upper()

            # 2. 고정 역할이라면 환경 내부 에이전트에게 주입
            if role_str != "RANDOM" and role_str in Role.__members__:
                role_enum = Role[role_str]

                # env.game.players는 게임 엔진 안의 'EnvAgent'들입니다.
                if i < len(env.game.players):
                    env.game.players[i].fixed_role = role_enum

    def build_env(self) -> MafiaEnv:
        """
        메인 프로세스용 환경
        [수정] Runner가 로그를 중앙 관리하므로, Env 내부에는 logger를 주지 않습니다.
        """
        encoder_map = {}
        if self.player_configs:
            for i, p_cfg in enumerate(self.player_configs):
                agent_type = p_cfg.get("type", "rl").lower()
                if agent_type == "rl":
                    bb = p_cfg.get("backbone", "mlp").lower()
                    encoder_map[i] = (
                        POMDPEncoder() if bb in ["lstm", "gru", "rnn"] else MDPEncoder()
                    )
                else:  # For 'rba', 'llm'
                    encoder_map[i] = AbsoluteEncoder()
        else:
            # Default to AbsoluteEncoder for all if no configs are provided (e.g., all RBA simulation)
            encoder_map = {
                i: AbsoluteEncoder() for i in range(config.game.PLAYER_COUNT)
            }

        env = MafiaEnv(encoder_map=encoder_map)
        self._inject_fixed_roles(env)

        return env

    def build_vec_env(self, num_envs: int = 8, num_cpus: int = 4):
        """
        병렬 학습 환경 생성
        """
        print(f"[System] Building Parallel Env: {num_envs} games with {num_cpus} CPUs")

        encoder_map = {}
        if self.player_configs:
            for i, p_cfg in enumerate(self.player_configs):
                agent_type = p_cfg.get("type", "rl").lower()
                if agent_type == "rl":
                    bb = p_cfg.get("backbone", "mlp").lower()
                    encoder_map[i] = (
                        POMDPEncoder() if bb in ["lstm", "gru", "rnn"] else MDPEncoder()
                    )
                else:  # For 'rba', 'llm'
                    encoder_map[i] = AbsoluteEncoder()
        else:
            # Default to MDPEncoder for RL-only parallel training
            encoder_map = {i: MDPEncoder() for i in range(config.game.PLAYER_COUNT)}

        env = MafiaEnv(encoder_map=encoder_map)

        self._inject_fixed_roles(env)

        # 2. PettingZoo -> Gymnasium 변환
        env = ss.pettingzoo_env_to_vec_env_v1(env)

        # 3. 병렬 연결 (num_cpus=0으로 단일 프로세스 모드 사용)
        try:
            vec_env = ss.concat_vec_envs_v1(
                env, num_vec_envs=num_envs, num_cpus=0, base_class="gymnasium"
            )
        except Exception as e:
            print(f"[Error] Parallel creation failed: {e}")
            print("[System] Switching to single process mode (Safe Mode)")
            vec_env = ss.concat_vec_envs_v1(
                env, num_vec_envs=num_envs, num_cpus=0, base_class="gymnasium"
            )

        return vec_env

    def build_agents(self) -> Dict[int, Any]:
        return AgentBuilder.build_agents(self.player_configs, logger=self.logger)

    def get_rl_agents(self, agents: Dict[int, Any]) -> Dict[int, Any]:
        return {i: a for i, a in agents.items() if isinstance(a, RLAgent)}

    def close(self):
        if self.logger:
            self.logger.close()
