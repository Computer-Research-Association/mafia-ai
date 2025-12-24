import torch
import torch.nn as nn
from torch.distributions import Categorical
import config
from ai.model import ActorCritic
from ai.buffer import RolloutBuffer

class PPO:
    def __init__(self, state_dim, action_dim):
        self.gamma = config.GAMMA
        self.eps_clip = config.EPS_CLIP
        self.k_epochs = config.K_EPOCHS
        self.lr = config.LR
        
        # 데이터 수집을 위한 버퍼 생성
        self.buffer = RolloutBuffer()
        
        # 현재 정책 (학습 대상)
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # 이전 정책 (Ratio 계산용, 가중치 고정)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        """
        상태를 받아 행동을 결정하고, 데이터(State, Action, LogProb)를 버퍼에 저장
        """
        with torch.no_grad():
            state = torch.FloatTensor(state)
            # action_probs, _ = self.policy_old(state) # Critic 값은 여기서 안 씀
            # policy_old를 사용하여 행동 결정 (학습 중인 policy가 아님에 주의)
            action_probs, _ = self.policy_old(state)
            
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
        # 버퍼에 저장
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        
        return action.item()

    def update(self):
        """
        모인 데이터를 바탕으로 PPO 업데이트 수행
        """
        # 1. Monte Carlo Estimate of Returns (Discounted Reward 계산)
        rewards = []
        discounted_reward = 0
        # 버퍼의 뒤에서부터 보상을 누적 계산
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # 텐서 변환 및 정규화 (학습 안정성)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # 버퍼에 있는 데이터를 텐서로 변환 (detach로 그래프 끊기)
        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach()
        
        # 2. K 에포크만큼 최적화 수행
        for _ in range(self.k_epochs):
            # 현재 정책으로 평가 (Evaluating old actions and values)
            action_probs, state_values = self.policy(old_states)
            
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            state_values = torch.squeeze(state_values)
            
            # Ratio 계산 (pi_theta / pi_theta_old)
            # log_prob끼리의 뺄셈 후 exp는 나눗셈과 같음
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Surrogate Loss 계산
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # 최종 Loss: Actor Loss + Critic Loss - Entropy Bonus
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # 역전파
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # 3. 학습된 정책을 Old Policy로 복사
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 4. 버퍼 비우기
        self.buffer.clear()