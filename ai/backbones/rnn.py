import torch.nn as nn

class RNNBackbone(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.feature_dim = hidden_dim
        
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0,
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

    def forward(self, x, hidden_state=None):
        # Returns: output, hidden_state
        return self.rnn(x, hidden_state)
