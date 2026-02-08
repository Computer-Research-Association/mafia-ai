import json
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from torch.distributions import Categorical
from core.managers.checkpoint import CheckpointManager

from core.agents.base_agent import BaseAgent
from ai.models.actor_critic import DynamicActorCritic
from ai.algorithms.ppo import PPO
from ai.algorithms.reinforce import REINFORCE
from config import config, Role
from core.engine.state import GameStatus, GameEvent, GameAction


class RLAgent(BaseAgent):
    """
    PPO, REINFORCE, RNN을 모두 지원하는 통합 RL 에이전트

    Multi-Discrete 액션 공간 지원: [Target, Role]

    Args:
        player_id: 플레이어 ID
        role: 플레이어 역할
        state_dim: 상태 벡터 차원
        action_dims: Multi-Discrete 액션 차원 리스트 [9, 5]
        algorithm: "ppo" 또는 "reinforce"
        backbone: "lstm", "gru"
        hidden_dim: 은닉층 차원
        num_layers: RNN 레이어 수
    """

    def __init__(
        self,
        player_id: int,
        role: Role,
        state_dim: int,
        action_dims: List[int] = [9, 5],  # [Target, Role]
        algorithm: str = "ppo",
        backbone: str = "mlp",  # Change default to mlp
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__(player_id, role)

        self.algorithm = algorithm.lower()
        self.backbone = backbone.lower()
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.action_dim = sum(action_dims)

        self.policy = DynamicActorCritic(
            state_dim=state_dim,
            action_dims=action_dims,
            backbone=backbone,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Move model to configured device (GPU or CPU)
        device = torch.device(config.train.DEVICE)
        self.policy = self.policy.to(device)

        # Check recurrence
        is_recurrent = self.backbone in ["lstm", "gru", "rnn"]

        if self.algorithm == "ppo":
            self.learner = PPO(
                model=self.policy, config=config, is_recurrent=is_recurrent
            )
        elif self.algorithm == "reinforce":
            self.learner = REINFORCE(
                model=self.policy, config=config, is_recurrent=is_recurrent
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.hidden_state = None
        self.current_action = None  # 현재 액션 저장

    def reset_hidden(self, batch_size: int = 1):
        """에피소드 시작 시 은닉 상태 초기화 (RNN Only) & 버퍼 리사이즈"""
        # 버퍼 리사이즈 (배치 크기가 변경될 수 있으므로 안전하게)
        if hasattr(self.learner, "buffer"):
            self.learner.buffer.resize(batch_size)

        if self.backbone in ["lstm", "gru"]:
            # 전달받은 batch_size를 정책 모델에 전달
            self.hidden_state = self.policy.init_hidden(batch_size=batch_size)
        else:
            # MLP 등
            self.hidden_state = None

    def reset_hidden_by_slot(self, slot_idx: int):
        """특정 슬롯의 은닉 상태만 초기화"""
        if self.hidden_state is not None:
            if isinstance(self.hidden_state, tuple):
                # LSTM: (h, c)
                self.hidden_state[0][:, slot_idx, :].fill_(0)
                self.hidden_state[1][:, slot_idx, :].fill_(0)
            else:
                # GRU: h
                self.hidden_state[:, slot_idx, :].fill_(0)

    def set_action(self, action: GameAction):
        """env.step()에서 호출되어 액션 설정"""
        self.current_action = action

    def get_action(self, status: GameStatus) -> GameAction:
        """
        BaseAgent 호환성을 위한 메서드 - MafiaAction 반환

        env.step()에서 설정한 current_action을 반환
        """
        if self.current_action is not None:
            return self.current_action

        # 폴백: PASS 액션
        return GameAction(target_id=-1, claim_role=None)

    def select_action_vector(
        self, state, action_mask: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        상태를 받아 Multi-Discrete 액션 벡터를 선택

        Args:
            state: 상태 벡터 또는 딕셔너리
            action_mask: 유효한 행동 마스크 (9, 5) 형태

        Returns:
            action_vector: [Target, Role] 형태의 리스트
        """
        obs_is_batch = False
        if isinstance(state, dict):
            obs = state["observation"]
            mask = state.get("action_mask", action_mask)
        else:
            obs = state
            mask = action_mask

        # Check if observation is batched (ndim > 1)
        if hasattr(obs, "ndim") and obs.ndim > 1:
            obs_is_batch = True
        elif isinstance(obs, list) and len(obs) > 0 and isinstance(obs[0], list):
            obs_is_batch = True

        state_dict = {"observation": obs, "action_mask": mask}

        # learner.select_action returns ([target, role], hidden_state)
        # Note: learner.select_action usually returns a batched list like [[t, r]] for 1 item batch
        action_vector, self.hidden_state = self.learner.select_action(
            state_dict, self.hidden_state
        )

        # If input was not batched (single item), unwrap the batch dimension from output
        if (
            not obs_is_batch
            and isinstance(action_vector, list)
            and len(action_vector) == 1
            and isinstance(action_vector[0], list)
        ):
            return action_vector[0]

        return action_vector

    def store_reward(self, reward: float, is_terminal: bool = False, slot_idx: int = 0):
        """보상 저장: 알고리즘별 버퍼 위치 확인"""
        if hasattr(self.learner, "buffer"):
            # PPO: 슬롯별 저장
            self.learner.buffer.insert_reward(slot_idx, reward, is_terminal)
        else:
            # REINFORCE 등: 단순 저장 (배치 미지원 가정 또는 추후 수정 필요)
            self.learner.rewards.append(reward)

    def update(self, expert_loader=None):
        """학습 수행 - 알고리즘 객체에 위임"""
        return self.learner.update(expert_loader=expert_loader)

    def save(self, filepath: str, extra_metadata: Dict[str, Any] = None):
        """
        CheckpointManager에게 저장을 위임
        """
        model_spec = {
            "algorithm": self.algorithm,
            "backbone": self.backbone,
            "state_dim": self.state_dim,
            "action_dims": self.action_dims,
            "structure": {
                "hidden_dim": self.policy.hidden_dim,
                "num_layers": self.policy.num_layers,
            },
        }

        state_dict = {
            "policy_state_dict": self.policy.state_dict(),
        }

        CheckpointManager.save_checkpoint(
            filepath=filepath,
            state_dict=state_dict,
            model_spec=model_spec,
            extra_metadata=extra_metadata,
        )

    def load(self, filepath: str):
        """
        CheckpointManager를 통해 로드
        """
        checkpoint = CheckpointManager.load_checkpoint(filepath)

        # 스펙 복원
        backbone = checkpoint.get("backbone", self.backbone)
        state_dim = checkpoint.get("state_dim", self.state_dim)
        action_dims = checkpoint.get("action_dims", self.action_dims)
        hidden_dim = checkpoint.get("hidden_dim", 128)
        num_layers = checkpoint.get("num_layers", 2)

        # 모델 재구축
        self.policy = DynamicActorCritic(
            state_dim=state_dim,
            action_dims=action_dims,
            backbone=backbone,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        
        # Move loaded model to configured device
        device = torch.device(config.train.DEVICE)
        self.policy = self.policy.to(device)

        # 상태 동기화
        self.backbone = backbone
        self.state_dim = state_dim
        self.action_dims = action_dims

        # 학습기 갱신
        if hasattr(self, "learner"):
            self.learner.policy = self.policy
            if self.algorithm == "ppo" and hasattr(self.learner, "policy_old"):
                self.learner.policy_old = DynamicActorCritic(
                    state_dim=state_dim,
                    action_dims=action_dims,
                    backbone=backbone,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                )
                self.learner.policy_old.load_state_dict(self.policy.state_dict())
                # Move policy_old to device as well
                self.learner.policy_old = self.learner.policy_old.to(device)
