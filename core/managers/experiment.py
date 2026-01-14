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
from core.agents.rl_agent import RLAgent
from core.agents.llm_agent import LLMAgent
from core.agents.rule_base_agent import RuleBaseAgent
from core.managers.logger import LogManager
from config import Role, config


class ExperimentManager:
    def __init__(self, args):
        self.args = args
        self.player_configs = getattr(args, "player_configs", [])
        self.mode = args.mode
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

        log_dir = str(getattr(self.args, "paths", {}).get("log_dir", "logs"))
        # 메인 프로세스는 정상적으로 파일을 씀
        return LogManager(
            experiment_name=experiment_name, log_dir=log_dir, write_mode=True
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
        env = MafiaEnv()
        self._inject_fixed_roles(env)

        return env

    def build_vec_env(self, num_envs: int = 8, num_cpus: int = 4):
        """
        병렬 학습 환경 생성
        """
        print(f"[System] Building Parallel Env: {num_envs} games with {num_cpus} CPUs")

        # 1. 단일 환경 템플릿 생성
        # [수정] 리스트 대신 단일 인스턴스를 사용해 에러 해결
        env = MafiaEnv()

        self._inject_fixed_roles(env)

        # 2. PettingZoo -> Gymnasium 변환
        env = ss.pettingzoo_env_to_vec_env_v1(env)

        # 3. 병렬 연결
        try:
            vec_env = ss.concat_vec_envs_v1(
                env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="gymnasium"
            )
        except Exception as e:
            print(f"[Error] Parallel creation failed: {e}")
            print("[System] Switching to single process mode (Safe Mode)")
            vec_env = ss.concat_vec_envs_v1(
                env, num_vec_envs=num_envs, num_cpus=0, base_class="gymnasium"
            )

        return vec_env

    def build_agents(self) -> Dict[int, Any]:
        state_dim = config.game.OBS_DIM
        agents = {}

        if not self.player_configs:
            # Fallback or error? main.py raised error.
            return {}

        for i, p_config in enumerate(self.player_configs):
            # 1. 설정에서 역할 문자열 가져오기
            role_str = p_config.get("role", "random").upper()

            # 2. 고정 역할 Enum 변환 (Random이면 None)
            fixed_role_enum = None
            if role_str != "RANDOM" and role_str in Role.__members__:
                fixed_role_enum = Role[role_str]

            # 3. [핵심 수정] 초기 역할(init_role)을 분기문 밖에서 미리 정의
            if fixed_role_enum is not None:
                init_role = fixed_role_enum
            else:
                init_role = Role.CITIZEN

            if p_config["type"] == "rl":
                init_role = fixed_role_enum if fixed_role_enum else Role.CITIZEN

                # 기본값은 GUI
                algo = p_config["algo"]
                backbone = p_config["backbone"]
                hidden_dim = p_config.get("hidden_dim", 128)
                num_layers = p_config.get("num_layers", 2)

                load_path = p_config.get("load_model_path")

                if load_path and os.path.exists(load_path):
                    checkpoint = torch.load(load_path, map_location="cpu")
                    if "backbone" in checkpoint:
                        backbone = checkpoint["backbone"]
                    if "hidden_dim" in checkpoint:
                        hidden_dim = checkpoint["hidden_dim"]
                    if "num_layers" in checkpoint:
                        num_layers = checkpoint["num_layers"]
                    if "algorithm" in checkpoint:
                        algo = checkpoint["algorithm"]

                agent = RLAgent(
                    player_id=i,
                    role=init_role,
                    state_dim=state_dim,
                    action_dims=config.game.ACTION_DIMS,
                    algorithm=algo,
                    backbone=backbone,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                )

                # 모델 경로
                if load_path:
                    if os.path.exists(load_path):
                        try:
                            agent.load(load_path)  # RLAgent의 load 메서드 호출
                            print(
                                f"[Experiment] Agent {i}: 모델 로드 성공 ({load_path})"
                            )
                        except Exception as e:
                            print(f"[Experiment] Agent {i}: 모델 로드 실패! ({e})")
                    else:
                        print(
                            f"[Experiment] Agent {i}: 경로에 파일이 없습니다 ({load_path})"
                        )

                agent.fixed_role = fixed_role_enum
                agents[i] = agent

            elif p_config["type"] == "llm":
                # ... (LLM 에이전트 생성 코드는 그대로 둠)
                agent = LLMAgent(player_id=i, logger=self.logger)
                agent.fixed_role = fixed_role_enum
                agents[i] = agent

            elif p_config["type"] == "rba":
                role_str = p_config.get("role", "citizen").upper()
                role = Role[role_str] if role_str in Role.__members__ else Role.CITIZEN

                agent = RuleBaseAgent(player_id=i, role=init_role)
                agent.fixed_role = fixed_role_enum
                agents[i] = agent
            else:
                pass

        return agents

    def get_rl_agents(self, agents: Dict[int, Any]) -> Dict[int, Any]:
        return {i: a for i, a in agents.items() if isinstance(a, RLAgent)}

    def close(self):
        if self.logger:
            self.logger.close()
