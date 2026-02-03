import torch
import torch.nn as nn
from ai.backbones import get_backbone


class DynamicActorCritic(nn.Module):
    """
    RNN(LSTM/GRU) and MLP based Actor-Critic model
    """

    def __init__(
        self,
        state_dim,
        action_dims=[9, 5],
        backbone="lstm",
        hidden_dim=128,
        num_layers=2,
    ):
        super(DynamicActorCritic, self).__init__()

        self.backbone_type = backbone.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.action_dims = action_dims

        # Create backbone using factory
        self.backbone = get_backbone(backbone, state_dim, hidden_dim, num_layers)

        feature_dim = self.backbone.feature_dim

        # Multi-Discrete Action Heads
        self.actor_target = nn.Linear(feature_dim, action_dims[0])
        self.actor_role = nn.Linear(feature_dim, action_dims[1])
        self.critic = nn.Linear(feature_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state, hidden_state=None):
        """
        Args:
            state: Input state (Batch, [Seq], Feature)
            hidden_state: RNN hidden state
        """
        # 1. Dimension adjustment (Batch, Seq, Feature)
        if state.dim() == 2:
            state = state.unsqueeze(1)

        # NOTE: Removed dangerous flattening logic for dim > 3 to prevent data mixing in parallel envs

        # Current Batch Size
        current_batch_size = state.size(0)

        # 2. Hidden state check and auto-init
        if self.backbone_type in ["lstm", "gru"]:
            if hidden_state is not None:
                if self.backbone_type == "lstm":
                    # LSTM: (h, c) tuple
                    h_size = hidden_state[0].size(1)
                else:
                    # GRU: h tensor
                    h_size = hidden_state.size(1)

                # Check batch size match
                if h_size != current_batch_size:
                    hidden_state = None

            if hidden_state is None:
                hidden_state = self.init_hidden(current_batch_size)

        # Backbone Forward
        features, new_hidden_state = self.backbone(state, hidden_state)

        # Multi-Head Output
        target_logits = self.actor_target(features)
        role_logits = self.actor_role(features)
        state_value = self.critic(features)

        return (target_logits, role_logits), state_value, new_hidden_state

    def init_hidden(self, batch_size=1):
        """Initialize RNN hidden state"""
        if self.backbone_type == "mlp":
            return None

        device = next(self.parameters()).device

        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

        if self.backbone_type == "lstm":
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return (h, c)
        else:  # gru
            return h
