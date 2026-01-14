import torch
import torch.optim as optim
from torch.distributions import Categorical

from ai.utils.buffer import RolloutBuffer

class REINFORCE:
    def __init__(self, model, config, is_recurrent=False):
        self.policy = model
        self.config = config
        self.is_recurrent = is_recurrent
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.train.LR)
        self.gamma = config.train.GAMMA
        self.max_grad_norm = config.train.MAX_GRAD_NORM
        
        self.buffer = RolloutBuffer()

    def select_action(self, state, hidden_state=None):
        if isinstance(state, dict):
            obs = state["observation"]
            mask = state.get("action_mask")
        else:
            obs = state
            mask = None

        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs)
            
            # Dimensions
            if self.is_recurrent:
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
                elif state_tensor.dim() == 2:
                    state_tensor = state_tensor.unsqueeze(1)
            else:
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)

            # Forward
            logits_tuple, _, new_hidden = self.policy(state_tensor, hidden_state)
            target_logits, role_logits = logits_tuple
            
            if target_logits.dim() > 2:
                target_logits = target_logits.squeeze(1)
                role_logits = role_logits.squeeze(1)

            # Masking
            if mask is not None:
                mask_tensor = torch.FloatTensor(mask)
                if mask_tensor.dim() == 1: mask_tensor = mask_tensor.unsqueeze(0)
                
                mask_target = mask_tensor[:, :9]
                mask_role = mask_tensor[:, 9:]
                
                target_logits_masked = target_logits.clone()
                role_logits_masked = role_logits.clone()
                
                target_logits_masked[mask_target == 0] = -1e9
                role_logits_masked[mask_role == 0] = -1e9
                
                target_logits = target_logits_masked
                role_logits = role_logits_masked
            
            dist_target = Categorical(logits=target_logits)
            dist_role = Categorical(logits=role_logits)
            
            action_target = dist_target.sample()
            action_role = dist_role.sample()
            
            log_prob = dist_target.log_prob(action_target) + dist_role.log_prob(action_role)
            
            action = torch.stack([action_target, action_role], dim=-1)
            
        self.buffer.states.append(torch.FloatTensor(obs))
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(log_prob)
        # Rewards and terminals added by agent
        
        return action.tolist(), new_hidden

    def update(self, expert_loader=None):
        episodes = self.buffer.get_episodes()
        
        total_loss = 0
        total_entropy = 0
        steps = 0
        
        for ep in episodes:
            # Returns Calculation
            R = 0
            returns = []
            for r in reversed(ep["rewards"]):
                 R = r + self.gamma * R
                 returns.insert(0, R)
            
            returns = torch.tensor(returns)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            # Re-evaluation for Gradients (BPTT style for RNN, standard for MLP)
            states = torch.stack(ep["states"]) # (Seq, Dim)
            actions = torch.stack(ep["actions"]).squeeze() # (Seq, 2)
            
            if self.is_recurrent:
                 # Add Batch Dim (1, Seq, Dim)
                 states_input = states.unsqueeze(0)
                 hidden = None # Start of episode
            else:
                 # Flatten for MLP? 
                 # REINFORCE update per episode is fine for MLP too.
                 states_input = states
                 hidden = None
            
            # Forward
            logits_tuple, _, _ = self.policy(states_input, hidden)
            target_logits, role_logits = logits_tuple
            
            # Formatting
            if self.is_recurrent:
                target_logits = target_logits.squeeze(0)
                role_logits = role_logits.squeeze(0)
            
            # Calculate Log Probs
            # Note: We are not masking here. Assuming actions taken were valid.
            
            dist_target = Categorical(logits=target_logits)
            dist_role = Categorical(logits=role_logits)
            
            if actions.dim() == 1: actions = actions.unsqueeze(0)

            log_prob = dist_target.log_prob(actions[:, 0]) + dist_role.log_prob(actions[:, 1])
            
            # Loss
            policy_loss = -log_prob * returns
            loss = policy_loss.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_entropy += (dist_target.entropy() + dist_role.entropy()).mean().item()
            steps += 1
        
        self.buffer.clear()
        
        return {
            "loss": total_loss / steps if steps > 0 else 0,
            "entropy": total_entropy / steps if steps > 0 else 0
        }
