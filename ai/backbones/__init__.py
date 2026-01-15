from .mlp import MLPBackbone
from .rnn import RNNBackbone

def get_backbone(name, input_dim, hidden_dim, num_layers=2):
    """
    Factory function to create backbone models.
    """
    name = name.lower()
    if name == "mlp":
        return MLPBackbone(input_dim, hidden_dim, num_layers)
    elif name in ["lstm", "gru"]:
        return RNNBackbone(name, input_dim, hidden_dim, num_layers)
    else:
        raise ValueError(f"Unknown backbone: {name}")
