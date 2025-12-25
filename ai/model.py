import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 더 깊은 공통 레이어 (128 → 128 → 64)
        # 시계열 정보를 더 잘 학습하기 위한 깊은 구조
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Actor Head (행동 확률)
        self.actor = nn.Linear(64, action_dim)
        
        # Critic Head (가치 판단)
        self.critic = nn.Linear(64, 1)
        
        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier/Glorot initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        # 깊은 특징 추출
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # Actor: 행동 확률 분포
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic: 상태 가치 평가
        state_value = self.critic(x)
        
        return action_probs, state_value