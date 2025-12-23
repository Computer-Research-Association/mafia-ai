import torch
import torch.optim as optim
from torch.distributions import Categorical
from ai.model import ActorCritic
import config

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim):
        # 우리가 만든 ActorCritic 모델 가져오기
        self.policy = ActorCritic(state_dim, action_dim)
        # Optimizer 연결 (config.py의 LR 사용)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.LR)
        self.gamma = config.GAMMA
        
        # REINFORCE용 간단한 메모리 (Buffer 클래스 없이 리스트로 처리)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """
        상태를 받아서 행동을 결정하고, 그 확률(log_prob)을 저장함
        """
        # 1. 텐서 변환 (배치 차원 추가: [State] -> [1, State])
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # 2. 모델 통과 (Critic 값은 REINFORCE에서 무시)
        probs, _ = self.policy(state)
        
        # 3. 확률 분포 생성 및 샘플링
        dist = Categorical(probs)
        action = dist.sample()
        
        # 4. 나중에 학습할 때 쓰려고 Log 확률 저장
        self.log_probs.append(dist.log_prob(action))
        
        return action.item()

    def update(self):
        """
        에피소드 하나가 끝나면 호출. 쌓인 데이터를 보고 학습 수행.
        """
        R = 0
        returns = []
        
        # 1. 뒤에서부터 보상 누적 (Discounted Reward) 계산
        # 예: [1, 0, 0] -> [1, 0.99, 0.98...] 처럼 미래 가치를 현재로 당겨옴
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        
        # (선택) 학습 안정화를 위한 정규화 (보상의 평균 0, 분산 1로 맞춤)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        # 2. 로스 계산 (Loss = -log_prob * Return)
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        # 3. 역전파 (Backpropagation)
        self.optimizer.zero_grad()
        # 모든 단계의 로스를 합쳐서 한 번에 업데이트
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # 4. 메모리 비우기 (다음 판을 위해)
        self.clear_memory()

    def clear_memory(self):
        del self.log_probs[:]
        del self.rewards[:]