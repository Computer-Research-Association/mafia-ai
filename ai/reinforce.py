import torch
import torch.optim as optim
from torch.distributions import Categorical
from ai.model import DynamicActorCritic
from config import config


class REINFORCE:
    def __init__(self, policy):
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.train.LR)
        self.gamma = config.train.GAMMA
        self.max_grad_norm = config.train.MAX_GRAD_NORM
        self.log_probs = []
        self.rewards = []
        self.states = []
        self.is_rnn = self.policy.backbone_type in ["lstm", "gru"]

    def select_action(self, state, hidden_state=None):
        if isinstance(state, dict):
            obs = state['observation']
            mask = state['action_mask']
        else:
            obs = state
            mask = None

        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0) if mask is not None else None
        
        probs, _, new_hidden = self.policy(state_tensor, hidden_state)
        
        if mask_tensor is not None:
            probs = probs * mask_tensor
            total_prob = probs.sum(dim=-1, keepdim=True)
            probs = probs / (total_prob + 1e-8)
        
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        self.states.append(state_tensor.squeeze(0))
        
        return action.item(), new_hidden

    def update(self, il_loss_fn=None):
        if len(self.rewards) == 0:
            return
            
        R = 0
        returns = []
        
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        loss = torch.stack(policy_loss).mean()
        
        if il_loss_fn is not None:
            old_states = torch.stack(self.states, dim=0).detach()
            il_loss = il_loss_fn(old_states)
            loss = loss + config.train.IL_COEF * il_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.clear_memory()

    def clear_memory(self):
        del self.log_probs[:]
        del self.rewards[:]
        del self.states[:]