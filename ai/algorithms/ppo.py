import torch
import torch.nn as nn
from torch.distributions import Categorical

from ai.models.actor_critic import DynamicActorCritic
from ai.utils.buffer import RolloutBuffer

class PPO:
    def __init__(self, model, config, is_recurrent=False):
        self.policy = model
        self.is_recurrent = is_recurrent
        self.config = config

        # Hyperparameters
        self.lr = config.train.LR
        self.gamma = config.train.GAMMA
        self.eps_clip = config.train.EPS_CLIP
        self.k_epochs = config.train.K_EPOCHS
        self.entropy_coef = config.train.ENTROPY_COEF
        self.value_loss_coef = config.train.VALUE_LOSS_COEF
        self.max_grad_norm = config.train.MAX_GRAD_NORM

        self.buffer = RolloutBuffer()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # Old policy for data collection
        self.policy_old = DynamicActorCritic(
            state_dim=model.state_dim,
            action_dims=model.action_dims,
            backbone=model.backbone_type,
            hidden_dim=model.hidden_dim,
            num_layers=model.num_layers,
        )
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, hidden_state=None):
        """
        Select action using policy_old.
        """
        if isinstance(state, dict):
            obs = state["observation"]
            mask = state.get("action_mask")
        else:
            obs = state
            mask = None

        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs)

            # Add dimensions for batch/seq if needed
            if self.is_recurrent:
                # (Batch=1, Seq=1, Feature)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
                elif state_tensor.dim() == 2:
                    state_tensor = state_tensor.unsqueeze(1)
            else:
                # (Batch=1, Feature)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)

            # Forward pass with policy_old
            logits_tuple, _, new_hidden = self.policy_old(state_tensor, hidden_state)
            target_logits, role_logits = logits_tuple

            # Squeeze if necessary to get (Batch, Dim)
            if target_logits.dim() > 2:
                target_logits = target_logits.squeeze(1)
                role_logits = role_logits.squeeze(1)

            # Masking
            if mask is not None:
                mask_tensor = torch.FloatTensor(mask)
                if mask_tensor.dim() == 1:
                    mask_tensor = mask_tensor.unsqueeze(0)

                mask_target = mask_tensor[:, :9]
                mask_role = mask_tensor[:, 9:]

                target_logits_masked = target_logits.clone()
                role_logits_masked = role_logits.clone()

                # Apply mask (-1e9 for invalid actions)
                target_logits_masked[mask_target == 0] = -1e9
                role_logits_masked[mask_role == 0] = -1e9
                
                target_logits = target_logits_masked
                role_logits = role_logits_masked

            # Distribution
            dist_target = Categorical(logits=target_logits)
            dist_role = Categorical(logits=role_logits)

            action_target = dist_target.sample()
            action_role = dist_role.sample()

            logprob_target = dist_target.log_prob(action_target)
            logprob_role = dist_role.log_prob(action_role)
            action_logprob = logprob_target + logprob_role

            action = torch.stack([action_target, action_role], dim=-1)

        # Save to buffer
        self.buffer.states.append(torch.FloatTensor(obs))
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        if mask is not None:
            self.buffer.masks.append(torch.FloatTensor(mask))

        return action.tolist(), new_hidden

    def _compute_gae(self, rewards, values, is_terminals):
        gae_lambda = 0.95
        advantages = []
        gae = 0
        
        # Basic GAE: delta = r + gamma * v_next - v
        # We assume value of next state after termination is 0
        values = list(values)
        values_next = values[1:] + [0]
        
        for r, v, v_next, done in zip(reversed(rewards), reversed(values), reversed(values_next), reversed(is_terminals)):
            if done:
                delta = r - v
                gae = delta
            else:
                delta = r + self.gamma * v_next - v
                gae = delta + self.gamma * gae_lambda * gae
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values, dtype=torch.float32)
        return returns, advantages

    def update(self):
        if self.is_recurrent:
            self._update_recurrent()
        else:
            self._update_mlp()
        
        # Updates done, sync policy_old
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def _update_mlp(self):
        # Flatten buffer data
        old_states = torch.stack(self.buffer.states).detach()
        old_actions = torch.stack(self.buffer.actions).detach().squeeze() 
        old_logprobs = torch.stack(self.buffer.logprobs).detach().squeeze()

        # Get masks
        old_masks = None
        if self.buffer.masks and self.buffer.masks[0] is not None:
             old_masks = torch.stack(self.buffer.masks).detach()

        # 1. Get Values for GAE
        with torch.no_grad():
            _, values, _ = self.policy(old_states)
            values = values.squeeze().tolist()
            if not isinstance(values, list): values = [values]

        # 2. Compute GAE
        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals
        returns, advantages = self._compute_gae(rewards, values, is_terminals)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Optimize for K epochs
        for _ in range(self.k_epochs):
            # Evaluate old actions and values (using CURRENT policy)
            # Flattened forward pass
            logits_tuple, state_values, _ = self.policy(old_states)
            target_logits, role_logits = logits_tuple
            
            state_values = state_values.squeeze()
            if target_logits.dim() > 2: target_logits = target_logits.squeeze(1); role_logits = role_logits.squeeze(1)

            # Masking
            if old_masks is not None:
                 mask_target = old_masks[:, :9]
                 mask_role = old_masks[:, 9:]
                 target_logits[mask_target==0] = -1e9
                 role_logits[mask_role==0] = -1e9

            dist_target = Categorical(logits=target_logits)
            dist_role = Categorical(logits=role_logits)

            action_target = old_actions[:, 0]
            action_role = old_actions[:, 1]
            
            logprobs = dist_target.log_prob(action_target) + dist_role.log_prob(action_role)
            dist_entropy = dist_target.entropy() + dist_role.entropy()

            # Ratios
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Advantages
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + self.value_loss_coef * self.MseLoss(state_values, returns) - self.entropy_coef * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def _update_recurrent(self):
        episodes = self.buffer.get_episodes()
        
        processed_episodes = []
        for ep in episodes:
            states = torch.stack(ep["states"]).detach() # (Seq, Dim)
            
            # Get values (Seq)
            with torch.no_grad():
                 # Input (1, Seq, Dim)
                 _, vals, _ = self.policy(states.unsqueeze(0), None)
                 vals = vals.squeeze().tolist() 
                 if not isinstance(vals, list): vals = [vals]
            
            rewards = ep["rewards"]
            is_terminals = ep["is_terminals"]
            
            returns, advantages = self._compute_gae(rewards, vals, is_terminals)
            
            ep["returns"] = returns
            ep["advantages"] = advantages
            processed_episodes.append(ep)

        # Optimization
        for _ in range(self.k_epochs):
            # Shuffle episodes if desired, but for RNN we just iterate
            # One update per episode (Batch Size = 1 Episode)
            for ep in processed_episodes:
                states = torch.stack(ep["states"]).detach() # (Seq, Dim)
                actions = torch.stack(ep["actions"]).detach().squeeze() # (Seq, 2)
                old_logprobs = torch.stack(ep["logprobs"]).detach().squeeze() # (Seq)
                
                returns = ep["returns"]
                advantages = ep["advantages"]
                
                masks = None
                if ep["masks"]:
                     masks = torch.stack(ep["masks"]).detach()

                # Forward with BPTT (Seq of states)
                # Input: (1, Seq, Dim)
                # Hidden: None (Start of episode)
                
                states_input = states.unsqueeze(0) # (1, Seq, Dim)
                
                # Policy Forward
                logits_tuple, state_values, _ = self.policy(states_input, None)
                target_logits, role_logits = logits_tuple
                
                # Output shapes: (1, Seq, Dim), (1, Seq, 1)
                target_logits = target_logits.squeeze(0) # (Seq, Dim)
                role_logits = role_logits.squeeze(0)
                state_values = state_values.squeeze(0).squeeze(-1) # (Seq)

                # Masking
                if masks is not None:
                    mask_target = masks[:, :9]
                    mask_role = masks[:, 9:]
                    target_logits[mask_target == 0] = -1e9
                    role_logits[role_mask == 0] = -1e9

                dist_target = Categorical(logits=target_logits)
                dist_role = Categorical(logits=role_logits)
                
                # Actions shape check
                if actions.dim() == 1: # If only 1 step
                    actions = actions.unsqueeze(0)

                logprobs = dist_target.log_prob(actions[:, 0]) + dist_role.log_prob(actions[:, 1])
                dist_entropy = dist_target.entropy() + dist_role.entropy()
                
                # Normalize advantages
                if len(advantages) > 1:
                     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

                # Ratios
                ratios = torch.exp(logprobs - old_logprobs)
                
                # Loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                
                loss = -torch.min(surr1, surr2) + self.value_loss_coef * self.MseLoss(state_values, returns) - self.entropy_coef * dist_entropy
                
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
