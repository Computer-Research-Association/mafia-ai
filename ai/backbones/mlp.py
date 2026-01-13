import torch.nn as nn

class MLPBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        layers = []
        
        # Input Layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden Layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        self.net = nn.Sequential(*layers)
        self.feature_dim = hidden_dim

    def forward(self, x, hidden_state=None):
        # MLP does not use hidden state, return None
        return self.net(x), None
