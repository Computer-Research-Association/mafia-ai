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
        self.is_rnn = policy.backbone_type in ["lstm", "gru"]

        if policy_old is None:
            self.policy_old = DynamicActorCritic(
                state_dim=policy.state_dim,
                action_dims=policy.action_dims,
                backbone=policy.backbone_type,
                hidden_dim=policy.hidden_dim,
                num_layers=policy.num_layers,
            )
            self.policy_old.load_state_dict(self.policy.state_dict())
        else:
            self.policy_old = policy_old

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, hidden_state=None):
        """
        상태를 입력받아 행동을 결정합니다. (배치 처리 지원)
        """
        if isinstance(state, dict):
            obs = state["observation"]
            mask = state.get("action_mask")
        else:
            obs = state
            mask = None

        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs)

            # RNN의 경우 시퀀스 차원 추가
            if self.is_rnn:
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
                elif state_tensor.dim() == 2:
                    state_tensor = state_tensor.unsqueeze(1)
            else:
                # MLP의 경우 배치 차원만 확인
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)

            # 모델 실행 - MLP일 경우 new_hidden은 None이 됨
            logits_tuple, _, new_hidden = self.policy_old(state_tensor, hidden_state)

            target_logits, role_logits = logits_tuple

            # Logits shape normalization: (Batch, Dim)
            if target_logits.dim() > 2: # Reduce from (Batch, 1, Dim) to (Batch, Dim)
                target_logits = target_logits.squeeze(1)
                role_logits = role_logits.squeeze(1)
            elif target_logits.dim() == 1 and state_tensor.size(0) > 1: # Batch case but 1D output
                 # This shouldn't happen with correct model forward, but for safety
                 pass
            elif target_logits.dim() == 1: # Single Sample
                pass 
            
            # Ensure shape is consistent for batch processing
            if target_logits.dim() == 1:
                 # If we had a single sample, temporarily view as batch of 1
                 # But dist.sample() handles 1D logits fine by returning scalar.
                 pass

            # 마스킹 적용 (Batch 지원)
            if mask is not None:
                mask_tensor = torch.FloatTensor(mask)
                if mask_tensor.dim() == 1:
                    mask_tensor = mask_tensor.unsqueeze(0)  # (1, 14)

                # 마스크 분리 [Target(9) | Role(5)]
                mask_target = mask_tensor[:, :9]
                mask_role = mask_tensor[:, 9:]

                # 마스킹 연산 (Broadcasting)
                valid_target = mask_target.sum(dim=1, keepdim=True) > 0
                valid_role = mask_role.sum(dim=1, keepdim=True) > 0

                target_logits_masked = target_logits.clone()
                role_logits_masked = role_logits.clone()
                
                # Careful with shapes here. mask_target is (B, 9), target_logits is (B, 9)
                target_logits_masked[mask_target == 0] = -1e9
                role_logits_masked[mask_role == 0] = -1e9
                
                # If all masked (which shouldn't happen with valid_target check), revert or handle
                # But here we rely on the mask logic being correct.
                target_logits = target_logits_masked
                role_logits = role_logits_masked

            # 확률 분포 생성 및 샘플링
            dist_target = Categorical(logits=target_logits)
            dist_role = Categorical(logits=role_logits)

            action_target = dist_target.sample()
            action_role = dist_role.sample()

            # 로그 확률 계산
            logprob_target = dist_target.log_prob(action_target)
            logprob_role = dist_role.log_prob(action_role)
            action_logprob = logprob_target + logprob_role

            # 행동 스택 (Batch, 2)
            action = torch.stack([action_target, action_role], dim=-1)

        # 버퍼 저장
        self.buffer.states.append(torch.FloatTensor(obs))  # Save original flattened obs
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.tolist(), new_hidden

    def update(self, expert_loader=None):
        if len(self.buffer.rewards) == 0:
            return {}

        expert_iter = iter(expert_loader) if expert_loader is not None else None

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)

        if len(rewards) > 1 and rewards.std() > 1e-7:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 병렬 환경에서는 buffer에 저장된 요소들이 (Batch, ...) 형태일 수 있음
        # 이를 (Time * Batch, ...) 형태로 Flatten 하거나 (Time, Batch, ...) 유지
        # 여기서는 간단하게 Stack 후 차원 정리

        # List[(Batch, Dim)] -> Tensor(Time, Batch, Dim)
        # Note: For MLP, we don't strictly need Time dim, but keeping it for compatibility
        # Or we can just cat everything to (Total_Batch, Dim) immediately.
        
        # Flattening strategy:
        # PPO usually treats (Time * Batch) as valid independent samples (for non-recurrent policy).
        # We just concat everything into one big batch.
        
        old_states = torch.cat([s.view(-1, s.size(-1)) for s in self.buffer.states], dim=0).detach()
        old_actions = torch.cat([a.view(-1, a.size(-1)) for a in self.buffer.actions], dim=0).detach()
        old_logprobs = torch.cat([l.view(-1) for l in self.buffer.logprobs], dim=0).detach()
        
        # Rewards are already flattened list, convert to tensor
        rewards = rewards.view(-1)

        # 30000판 처리를 위해 데이터를 단순히 플래튼(Flatten)하여 학습 (RNN 시퀀스 무시)
        # MLP는 시퀀스 차원이 없으므로 바로 플래튼된 상태로 사용 가능
        
        # 데이터셋 생성
        dataset_size = old_states.size(0)
        indices = torch.arange(dataset_size)

        avg_loss_all_epochs = 0
        avg_entropy_all_epochs = 0
        avg_kl_all_epochs = 0
        avg_clip_frac_all_epochs = 0

        for _ in range(self.k_epochs):
            # Shuffle indices
            indices = indices[torch.randperm(dataset_size)]

            # Mini-batch loop (Batch size 예를 들어 64 or 256)
            batch_size = 256
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_idx = indices[start_idx:end_idx]

                mb_states = old_states[batch_idx]  # (B, Dim)
                mb_actions = old_actions[batch_idx]  # (B, 2)
                mb_old_logprobs = old_logprobs[batch_idx]
                mb_rewards = rewards[batch_idx]

                # Forward (RNN hidden state is lost in simple batching - limitation)
                # RNN을 쓰더라도 여기서는 hidden=None으로 초기화하여 단기 기억만 사용하거나,
                # 전체 시퀀스를 유지해야 하는데 코드가 복잡해짐. 우선 실행을 위해 hidden=None 처리.
                # Model forward accepts (Batch, Dim) -> converts to (Batch, 1, Dim)
                logits_tuple, state_values, _ = self.policy(mb_states)

                target_logits, role_logits = logits_tuple

                # (Batch, 1, Dim) -> (Batch, Dim)
                if target_logits.dim() == 3:
                    target_logits = target_logits.squeeze(1)
                    role_logits = role_logits.squeeze(1)

                state_values = state_values.view(-1)

                dist_target = Categorical(logits=target_logits)
                dist_role = Categorical(logits=role_logits)

                logprobs_target = dist_target.log_prob(mb_actions[:, 0])
                logprobs_role = dist_role.log_prob(mb_actions[:, 1])
                logprobs = logprobs_target + logprobs_role

                dist_entropy = dist_target.entropy() + dist_role.entropy()

                ratios = torch.exp(logprobs - mb_old_logprobs)

                # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = logprobs - mb_old_logprobs
                    approx_kl = ((ratios - 1) - log_ratio).mean()
                    clip_fracs = ((ratios - 1.0).abs() > self.eps_clip).float().mean()

                advantages = mb_rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
                )

                loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * self.MseLoss(state_values, mb_rewards)
                    - self.entropy_coef * dist_entropy
                )

                # === Imitation Learning Loss Injection ===
                if expert_iter is not None:
                    try:
                        batch = next(expert_iter)
                    except StopIteration:
                        expert_iter = iter(expert_loader)
                        batch = next(expert_iter)
                    
                    if batch is not None:
                        exp_obs = batch['obs']
                        exp_target = batch['act_target']
                        exp_role = batch['act_role']
                        
                        # Forward pass for expert data
                        logits_tuple, _, _ = self.policy(exp_obs)
                        target_logits, role_logits = logits_tuple

                        if target_logits.dim() == 3:
                            target_logits = target_logits.squeeze(1)
                            role_logits = role_logits.squeeze(1)
                        
                        # CrossEntropyLoss
                        ce_loss = nn.CrossEntropyLoss()
                        il_loss = ce_loss(target_logits, exp_target) + ce_loss(role_logits, exp_role)
                        
                        loss += config.train.IL_COEF * il_loss

                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                avg_loss_all_epochs += loss.mean().item()
                avg_entropy_all_epochs += dist_entropy.mean().item()
                avg_kl_all_epochs += approx_kl.item()
                avg_clip_frac_all_epochs += clip_fracs.item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        # 총 배치가 몇 번 돌았는지 계산
        total_batches = self.k_epochs * (dataset_size // batch_size + 1)

        return {
            "loss": avg_loss_all_epochs / total_batches if total_batches > 0 else 0,
            "entropy": (
                avg_entropy_all_epochs / total_batches if total_batches > 0 else 0
            ),
            "approx_kl": (
                avg_kl_all_epochs / total_batches if total_batches > 0 else 0
            ),
            "clip_frac": (
                avg_clip_frac_all_epochs / total_batches if total_batches > 0 else 0
            ),
        }

    def _split_episodes(self, data_list, is_terminals):
        # (Not used in batch update mode)
        pass
