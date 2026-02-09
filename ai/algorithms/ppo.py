import torch
import torch.nn as nn
import numpy as np
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
        self.policy_old.to(next(self.policy.parameters()).device)

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

        obs_is_batch = False
        if hasattr(obs, "ndim") and obs.ndim > 1:
            obs_is_batch = True
        elif isinstance(obs, list) and len(obs) > 0 and isinstance(obs[0], list):
            obs_is_batch = True

        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).to(
                next(self.policy.parameters()).device
            )

            # Add dimensions for batch/seq if needed
            if self.is_recurrent:
                # (Batch, Seq=1, Feature)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
                elif state_tensor.dim() == 2:
                    state_tensor = state_tensor.unsqueeze(1)
            else:
                # (Batch, Feature)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)

            # Forward pass with policy_old
            logits_tuple, state_values, new_hidden = self.policy_old(
                state_tensor, hidden_state
            )
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

        # Save to buffer per slot
        actions_list = action.detach().cpu()
        logprobs_list = action_logprob.detach().cpu()
        values_list = state_values.detach().cpu().squeeze()
        if values_list.dim() == 0:
            values_list = values_list.unsqueeze(0)

        # Batch iteration
        batch_size = len(obs) if obs_is_batch else 1

        # Ensure lists if not batch
        if not obs_is_batch:
            obs = [obs]
            if mask is not None:
                mask = [mask]

        for i in range(batch_size):
            # Extract slice of hidden state for this slot if available
            slot_hidden = None
            if hidden_state is not None:
                if isinstance(hidden_state, tuple):  # LSTM
                    h, c = hidden_state
                    # Slicing (Layers, Batch, Dim) -> (Layers, 1, Dim)
                    # Use clone to detach from graph but keep value? No, just value storage.
                    # We detach because buffer storage is for restarting BPTT, not backpropping through previous batch.
                    slot_h = h[:, i : i + 1, :].detach().clone()
                    slot_c = c[:, i : i + 1, :].detach().clone()
                    slot_hidden = (slot_h, slot_c)
                else:  # GRU
                    slot_hidden = hidden_state[:, i : i + 1, :].detach().clone()

            self.buffer.insert(
                slot_idx=i,
                state=obs[i],
                action=actions_list[i],
                logprob=logprobs_list[i],
                mask=mask[i] if mask is not None else None,
                value=values_list[i],
                hidden_state=slot_hidden,
            )

        return action.tolist(), new_hidden

    def update(self, expert_loader=None):
        # Flattened data components
        flat_states = []
        flat_actions = []
        flat_logprobs = []
        flat_returns = []
        flat_advantages = []
        flat_masks = []

        # Helper for GAE
        gae_lambda = getattr(self.config.train, "GAE_LAMBDA", 0.95)

        # Process each slot (trajectory)
        for i in range(self.buffer.num_envs):
            # Get data for this slot
            rewards = self.buffer.rewards[i]
            if not rewards:
                continue

            values = self.buffer.values[i]
            terminals = self.buffer.is_terminals[i]

            # Compute GAE
            advantages = []
            gae = 0
            # We assume value of next state after termination is 0
            # Append 0 to values to handle index i+1
            values_ext = values + [0]

            for t in reversed(range(len(rewards))):
                if terminals[t]:
                    delta = rewards[t] - values[t]
                    gae = delta
                else:
                    delta = rewards[t] + self.gamma * values_ext[t + 1] - values[t]
                    gae = delta + self.gamma * gae_lambda * gae
                advantages.append(gae)

            advantages.reverse()
                
            # Returns = Adv + Value
            returns = [a + v for a, v in zip(advantages, values)]

            # Store processed data
            flat_states.append(torch.tensor(np.array(self.buffer.states[i]), dtype=torch.float32))
            flat_actions.append(torch.stack(self.buffer.actions[i]))
            flat_logprobs.append(torch.stack(self.buffer.logprobs[i]))
            flat_returns.append(torch.tensor(np.array(returns), dtype=torch.float32))
            flat_advantages.append(torch.tensor(np.array(advantages), dtype=torch.float32))
            
            if self.buffer.masks[i]:
                # Numpy list -> Numpy Array -> Tensor conversion
                mask_seq = np.array(self.buffer.masks[i])
                flat_masks.append(torch.tensor(mask_seq, dtype=torch.float32))
            else:
                pass  # Handle later

        if not flat_states:
            return

        if self.is_recurrent:
            # Retrieve initial hidden states for each slot
            initial_hiddens = self.buffer.initial_hidden_states
            metrics = self._update_recurrent_opt(
                flat_states,
                flat_actions,
                flat_logprobs,
                flat_returns,
                flat_advantages,
                flat_masks,
                initial_hiddens,
                expert_loader=expert_loader,
            )
        else:
            metrics = self._update_mlp_opt(
                flat_states,
                flat_actions,
                flat_logprobs,
                flat_returns,
                flat_advantages,
                flat_masks,
                expert_loader=expert_loader,
            )

        # Updates done, sync policy_old
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return metrics

    def _update_mlp_opt(
        self, states, actions, logprobs, returns, advantages, masks, expert_loader=None
    ):
        device = next(self.policy.parameters()).device

        # Flatten all slots
        b_states = torch.cat(states).to(device)
        b_actions = torch.cat(actions).to(device)
        b_logprobs = torch.cat(logprobs).to(device)
        b_returns = torch.cat(returns).to(device)
        b_advantages = torch.cat(advantages).to(device)

        b_masks = None
        if masks:
            b_masks = torch.cat(masks).to(device)

        # Normalize advantages
        if b_advantages.numel() > 1:
            b_advantages = (b_advantages - b_advantages.mean()) / (
                b_advantages.std() + 1e-7
            )

        total_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        total_clip_frac = 0
        steps = 0

        # Optimize
        for _ in range(self.k_epochs):
            # Forward (Batch)
            logits_tuple, state_values, _ = self.policy(b_states)
            target_logits, role_logits = logits_tuple

            state_values = state_values.squeeze()
            if target_logits.dim() > 2:
                target_logits = target_logits.squeeze(1)
                role_logits = role_logits.squeeze(1)

            # Masking
            if b_masks is not None:
                mask_target = b_masks[:, :9]
                mask_role = b_masks[:, 9:]
                target_logits[mask_target == 0] = -1e9
                role_logits[mask_role == 0] = -1e9

            dist_target = Categorical(logits=target_logits)
            dist_role = Categorical(logits=role_logits)

            action_target = b_actions[:, 0]
            action_role = b_actions[:, 1]

            logprobs_new = dist_target.log_prob(action_target) + dist_role.log_prob(
                action_role
            )
            dist_entropy = dist_target.entropy() + dist_role.entropy()

            # Ratios
            ratios = torch.exp(logprobs_new - b_logprobs)

            # KL Divergence, Clip Fraction
            with torch.no_grad():
                approx_kl = (b_logprobs - logprobs_new).mean().item()
                clip_frac = (
                    (torch.abs(ratios - 1.0) > self.eps_clip).float().mean().item()
                )

            # Loss
            surr1 = ratios * b_advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
            )

            loss = (
                -torch.min(surr1, surr2)
                + self.value_loss_coef * self.MseLoss(state_values, b_returns)
                - self.entropy_coef * dist_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            total_loss += loss.mean().item()
            total_entropy += dist_entropy.mean().item()
            total_approx_kl += approx_kl
            total_clip_frac += clip_frac
            steps += 1

        return {
            "loss": total_loss / steps if steps > 0 else 0,
            "entropy": total_entropy / steps if steps > 0 else 0,
            "approx_kl": total_approx_kl / steps if steps > 0 else 0,
            "clip_frac": total_clip_frac / steps if steps > 0 else 0,
        }

    def _update_recurrent_opt(
        self,
        states,
        actions,
        logprobs,
        returns,
        advantages,
        masks,
        initial_hiddens=None,
        expert_loader=None,
    ):
        device = next(self.policy.parameters()).device

        # 1. Global Advantage Normalization
        # Move all advantages to device for calculation
        advantages = [adv.to(device) for adv in advantages]
        all_advantages = torch.cat(advantages)

        if all_advantages.numel() > 1:
            adv_mean = all_advantages.mean()
            adv_std = all_advantages.std() + 1e-7
            # Normalize list elements
            advantages = [(adv - adv_mean) / adv_std for adv in advantages]

        total_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        total_clip_frac = 0
        steps = 0

        # Create indices for shuffling
        num_trajectories = len(states)
        indices = np.arange(num_trajectories)

        # 2. Dataset-level Epochs (Standard PPO Loop)
        for _ in range(self.k_epochs):
            # Shuffle trajectories for each epoch to break correlation
            np.random.shuffle(indices)

            for i in indices:
                seq_states = states[i].to(device)  # (Seq, Dim)
                seq_actions = actions[i].to(device)
                seq_logprobs = logprobs[i].to(device)
                seq_returns = returns[i].to(device)
                seq_advantages = advantages[i]  # Already on device and normalized
                seq_masks = masks[i].to(device) if masks else None

                # Get initial hidden state for this slot
                start_hidden = initial_hiddens[i] if initial_hiddens else None
                if start_hidden is not None:
                    if isinstance(start_hidden, tuple):
                        start_hidden = (
                            start_hidden[0].to(device),
                            start_hidden[1].to(device),
                        )
                    else:
                        start_hidden = start_hidden.to(device)

                # Forward (Batch=1, Seq, Dim)
                states_input = seq_states.unsqueeze(0)

                # Pass start_hidden (which is detached snapshot)
                logits_tuple, state_values, _ = self.policy(states_input, start_hidden)
                target_logits, role_logits = logits_tuple

                target_logits = target_logits.squeeze(0)
                role_logits = role_logits.squeeze(0)
                state_values = state_values.squeeze(0).squeeze(-1)

                if seq_masks is not None:
                    mask_target = seq_masks[:, :9]
                    mask_role = seq_masks[:, 9:]
                    target_logits[mask_target == 0] = -1e9
                    role_logits[mask_role == 0] = -1e9

                dist_target = Categorical(logits=target_logits)
                dist_role = Categorical(logits=role_logits)

                act_target = seq_actions[:, 0]
                act_role = seq_actions[:, 1]

                logprobs_new = dist_target.log_prob(act_target) + dist_role.log_prob(
                    act_role
                )
                dist_entropy = dist_target.entropy() + dist_role.entropy()

                ratios = torch.exp(logprobs_new - seq_logprobs.detach())

                # KL Divergence, Clip Fraction
                with torch.no_grad():
                    approx_kl = (seq_logprobs.detach() - logprobs_new).mean().item()
                    clip_frac = (
                        (torch.abs(ratios - 1.0) > self.eps_clip).float().mean().item()
                    )

                surr1 = ratios * seq_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * seq_advantages
                )

                loss = (
                    -torch.min(surr1, surr2)
                    + self.value_loss_coef * self.MseLoss(state_values, seq_returns)
                    - self.entropy_coef * dist_entropy
                )

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                total_loss += loss.mean().item()
                total_entropy += dist_entropy.mean().item()
                total_approx_kl += approx_kl
                total_clip_frac += clip_frac
                steps += 1

        return {
            "loss": total_loss / steps if steps > 0 else 0,
            "entropy": total_entropy / steps if steps > 0 else 0,
            "approx_kl": total_approx_kl / steps if steps > 0 else 0,
            "clip_frac": total_clip_frac / steps if steps > 0 else 0,
        }
