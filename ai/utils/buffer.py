import torch

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.hidden_states = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden_states[:]

    def get_episodes(self):
        """
        Splits the buffer into episodic sequences based on is_terminals.
        Returns a list of dictionaries, each containing data for one episode.
        """
        episodes = []
        if len(self.is_terminals) == 0:
            return episodes

        start_idx = 0
        for i, done in enumerate(self.is_terminals):
            if done:
                # Extract slice for this episode
                episode_data = {
                    "states": self.states[start_idx : i+1],
                    "actions": self.actions[start_idx : i+1],
                    "logprobs": self.logprobs[start_idx : i+1],
                    "rewards": self.rewards[start_idx : i+1],
                    "is_terminals": self.is_terminals[start_idx : i+1],
                }
                episodes.append(episode_data)
                start_idx = i + 1
        
        # If there is remaining data that didn't end with done (e.g. truncated), add it
        if start_idx < len(self.is_terminals):
             episode_data = {
                "states": self.states[start_idx:],
                "actions": self.actions[start_idx:],
                "logprobs": self.logprobs[start_idx:],
                "rewards": self.rewards[start_idx:],
                "is_terminals": self.is_terminals[start_idx:],
            }
             episodes.append(episode_data)

        return episodes
