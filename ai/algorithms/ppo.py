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
                # Valid check might be needed to avoid all -Inf
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

        return action.tolist(), new_hidden

    def update(self):
        if self.is_recurrent:
            self._update_recurrent()
        else:
            self._update_mlp()
        
        # Updates done, sync policy_old
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def _update_mlp(self):
        # Calculate rewards (Monte Carlo)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        if len(rewards) > 1 and rewards.std() > 1e-7:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Flatten buffer data
        old_states = torch.stack(self.buffer.states).detach()
        old_actions = torch.stack(self.buffer.actions).detach().squeeze() 
        old_logprobs = torch.stack(self.buffer.logprobs).detach().squeeze()

        # Optimize for K epochs
        for _ in range(self.k_epochs):
            # Evaluate old actions and values (using CURRENT policy)
            # Flattened forward pass
            logits_tuple, state_values, _ = self.policy(old_states)
            target_logits, role_logits = logits_tuple
            
            # Re-Calculate LogProbs
            # We need to apply masking if we want exact same probs, but states stored don't have masks attached in buffer directly usually?
            # Existing PPO code didn't store masks in buffer. It just learned from pure logits?
            # Wait, if we don't mask during update, we might assign probability to invalid actions? 
            # But the actions taken (old_actions) ARE valid.
            # However, Categorical distribution normalization depends on all logits.
            # If we don't mask, the dist is different.
            # Original code: `mb_states = old_states[batch_idx]`, passed to `self.policy`.
            # Original code `update` method DID NOT apply masks.
            # This is a common simplification/oversight in simple PPO impls if masks aren't stored.
            # I will follow original implementation (no masks in update).
            
            # Squeeze outputs
            state_values = state_values.squeeze()
            if target_logits.dim() > 2: target_logits = target_logits.squeeze(1); role_logits = role_logits.squeeze(1)

            dist_target = Categorical(logits=target_logits)
            dist_role = Categorical(logits=role_logits)

            action_target = old_actions[:, 0]
            action_role = old_actions[:, 1]
            
            logprobs = dist_target.log_prob(action_target) + dist_role.log_prob(action_role)
            dist_entropy = dist_target.entropy() + dist_role.entropy()

            # Ratios
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Advantages
            advantages = rewards - state_values.detach()

            # Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + self.value_loss_coef * self.MseLoss(state_values, rewards) - self.entropy_coef * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def _update_recurrent(self):
        episodes = self.buffer.get_episodes()
        
        # Pre-calculate rewards for all episodes
        # It's better to do this per episode
        processed_episodes = []
        for ep in episodes:
            rewards = []
            discounted_reward = 0
            for r, t in zip(reversed(ep["rewards"]), reversed(ep["is_terminals"])):
                if t: discounted_reward = 0
                discounted_reward = r + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            
            rewards = torch.tensor(rewards, dtype=torch.float32)
            if len(rewards) > 1 and rewards.std() > 1e-7:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            
            ep["discounted_rewards"] = rewards
            processed_episodes.append(ep)

        # Optimization
        for _ in range(self.k_epochs):
            # Shuffle episodes if desired, but for RNN we just iterate
            # One update per episode (Batch Size = 1 Episode)
            for ep in processed_episodes:
                states = torch.stack(ep["states"]).detach() # (Seq, Dim)
                actions = torch.stack(ep["actions"]).detach().squeeze() # (Seq, 2)
                old_logprobs = torch.stack(ep["logprobs"]).detach().squeeze() # (Seq)
                rewards = ep["discounted_rewards"]

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

                dist_target = Categorical(logits=target_logits)
                dist_role = Categorical(logits=role_logits)
                
                # Actions shape check
                if actions.dim() == 1: # If only 1 step
                    actions = actions.unsqueeze(0)

                logprobs = dist_target.log_prob(actions[:, 0]) + dist_role.log_prob(actions[:, 1])
                dist_entropy = dist_target.entropy() + dist_role.entropy()
                
                # Ratios
                ratios = torch.exp(logprobs - old_logprobs)
                
                # Advantages
                advantages = rewards - state_values.detach()
                
                # Loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                
                loss = -torch.min(surr1, surr2) + self.value_loss_coef * self.MseLoss(state_values, rewards) - self.entropy_coef * dist_entropy
                
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
