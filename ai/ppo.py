import torch
import torch.nn as nn
from torch.distributions import Categorical
from config import config
from ai.model import DynamicActorCritic
from ai.buffer import RolloutBuffer


class PPO:
    def __init__(self, policy, policy_old=None):
        self.gamma = config.train.GAMMA
        self.eps_clip = config.train.EPS_CLIP
        self.k_epochs = config.train.K_EPOCHS
        self.lr = config.train.LR
        self.entropy_coef = config.train.ENTROPY_COEF
        self.value_loss_coef = config.train.VALUE_LOSS_COEF
        self.max_grad_norm = config.train.MAX_GRAD_NORM
        
        self.buffer = RolloutBuffer()
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        if policy_old is None:
            self.policy_old = DynamicActorCritic(
                state_dim=policy.state_dim,
                action_dim=policy.actor.out_features,
                backbone=policy.backbone_type,
                hidden_dim=policy.hidden_dim,
                num_layers=policy.num_layers
            )
            self.policy_old.load_state_dict(self.policy.state_dict())
        else:
            self.policy_old = policy_old
            
        self.MseLoss = nn.MSELoss()
        self.is_rnn = self.policy.backbone_type in ["lstm", "gru"]

    def select_action(self, state, hidden_state=None):
        if isinstance(state, dict):
            obs = state['observation']
            mask = state['action_mask']  # Shape: (3, 9, 5)
        else:
            obs = state
            mask = None
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # logits_tuple: (type_logits, target_logits, role_logits)
            logits_tuple, _, new_hidden = self.policy_old(state_tensor, hidden_state)
            
            # Unpack logits
            target_logits, role_logits = logits_tuple
            target_logits = target_logits.squeeze(0)
            role_logits = role_logits.squeeze(0)
            
            # Apply Masking if available
            if mask is not None:
                mask_tensor = torch.FloatTensor(mask)
                # mask shape: (14,) -> [Target(9), Role(5)]
                mask_target = mask_tensor[:9]
                mask_role = mask_tensor[9:]
                
                # Apply mask (set logits to -inf for invalid actions)
                target_logits = target_logits.masked_fill(mask_target == 0, -1e9)
                role_logits = role_logits.masked_fill(mask_role == 0, -1e9)

            # Create distributions
            dist_target = Categorical(logits=target_logits)
            dist_role = Categorical(logits=role_logits)
            
            # Sample actions
            action_target = dist_target.sample()
            action_role = dist_role.sample()
            
            # Calculate log probs
            logprob_target = dist_target.log_prob(action_target)
            logprob_role = dist_role.log_prob(action_role)
            
            # Total log prob (sum of independent log probs)
            action_logprob = logprob_target + logprob_role
            
            action = torch.stack([action_target, action_role])
            
        self.buffer.states.append(state_tensor.squeeze(0))
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        if self.is_rnn and new_hidden is not None:
            self.buffer.hidden_states.append(new_hidden)
        
        return action.tolist(), new_hidden

    def update(self, il_loss_fn=None):
        if len(self.buffer.rewards) == 0:
            return
            
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32)
        if rewards.std() > 1e-7:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach() # Shape: (N, 2)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach()
        
        for _ in range(self.k_epochs):
            if self.is_rnn:
                # RNN: 에피소드 경계를 고려한 시퀀스 처리
                episodes = self._split_episodes(old_states, self.buffer.is_terminals)
                pass 
            
            # Forward pass
            logits_tuple, state_values, _ = self.policy(old_states)
            target_logits, role_logits = logits_tuple
            
            # Create distributions
            dist_target = Categorical(logits=target_logits)
            dist_role = Categorical(logits=role_logits)
            
            # Get log probs for old actions
            # old_actions: (N, 2) -> target, role
            logprobs_target = dist_target.log_prob(old_actions[:, 0])
            logprobs_role = dist_role.log_prob(old_actions[:, 1])
            
            logprobs = logprobs_target + logprobs_role
            dist_entropy = dist_target.entropy() + dist_role.entropy()
            state_values = state_values.squeeze()
            
            # Ratios
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.entropy_coef * dist_entropy
            
            # IL Loss
            if il_loss_fn is not None:
                loss = loss.mean() + config.train.IL_COEF * il_loss_fn(old_states)
            else:
                loss = loss.mean()
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
                    if len(ep_states) == 0:
                        continue
                    ep_states_3d = ep_states.unsqueeze(0)
                    action_probs, state_values, _ = self.policy(ep_states_3d)
                    all_action_probs.append(action_probs.squeeze(0))
                    all_state_values.append(state_values.squeeze(0))
                
                action_probs = torch.cat(all_action_probs, dim=0)
                state_values = torch.cat(all_state_values, dim=0)
            else:
                action_probs, state_values, _ = self.policy(old_states)
            
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs)
            
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2)
            value_loss = self.MseLoss(state_values, rewards)
            entropy_loss = -dist_entropy
            
            loss = actor_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            if il_loss_fn is not None:
                il_loss = il_loss_fn(old_states)
                loss = loss + config.train.IL_COEF * il_loss
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def _split_episodes(self, states, is_terminals):
        """에피소드 경계에 따라 상태를 분리"""
        episodes = []
        current_episode = []
        
        for i, state in enumerate(states):
            current_episode.append(state)
            if is_terminals[i]:
                if current_episode:
                    episodes.append(torch.stack(current_episode))
                current_episode = []
        
        if current_episode:
            episodes.append(torch.stack(current_episode))
        
        return episodes