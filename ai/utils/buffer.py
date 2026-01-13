import torch

class RolloutBuffer:
    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.reset()
    
    def reset(self):
        self.states = [[] for _ in range(self.num_envs)]
        self.actions = [[] for _ in range(self.num_envs)]
        self.logprobs = [[] for _ in range(self.num_envs)]
        self.rewards = [[] for _ in range(self.num_envs)]
        self.is_terminals = [[] for _ in range(self.num_envs)]
        self.masks = [[] for _ in range(self.num_envs)]
        self.values = [[] for _ in range(self.num_envs)]

    def resize(self, num_envs):
        self.num_envs = num_envs
        self.reset()

    def clear(self):
        self.reset()

    def insert(self, slot_idx, state, action, logprob, mask=None, value=None):
        if slot_idx >= self.num_envs:
            return
        self.states[slot_idx].append(state)
        self.actions[slot_idx].append(action)
        self.logprobs[slot_idx].append(logprob)
        if mask is not None:
            self.masks[slot_idx].append(mask)
        if value is not None:
            self.values[slot_idx].append(value)

    def insert_reward(self, slot_idx, reward, is_terminal):
        if slot_idx >= self.num_envs:
            return
        self.rewards[slot_idx].append(reward)
        self.is_terminals[slot_idx].append(is_terminal)

    def get_data(self):
        """
        Returns data appropriate for training.
        For RNN: List of trajectories (sequences).
        For MLP: Flattened batch.
        But logic is better handled in PPO update method using raw lists.
        """
        return self
