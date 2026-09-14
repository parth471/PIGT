import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """
    Transformer encoder operating across the temporal dimension.

    Input:
        [batch, time, nodes, hidden]

    Output:
        [batch, nodes, hidden]

    The Transformer attends only across the 7 historical
    timesteps, not across all 12,204 nodes.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_time_steps: int = 7,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_time_steps = max_time_steps

        # Learnable temporal positional embeddings.
        self.time_embedding = nn.Parameter(
            torch.zeros(
                1,
                max_time_steps,
                1,
                hidden_dim,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape:
                [B, T, N, H]

        Returns
        -------
        torch.Tensor
            Shape:
                [B, N, H]
        """

        batch_size, time_steps, num_nodes, hidden_dim = x.shape

        if time_steps > self.max_time_steps:
            raise ValueError(
                f"Received {time_steps} timesteps, "
                f"but max_time_steps={self.max_time_steps}"
            )

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dimension {self.hidden_dim}, "
                f"got {hidden_dim}"
            )

        # Add temporal positional information.
        x = x + self.time_embedding[:, :time_steps]

        # Rearrange so each ocean node has its own 7-step sequence:
        #
        # [B, T, N, H]
        #       ↓
        # [B*N, T, H]
        #
        # The Transformer therefore attends over TIME only.
        x = x.permute(0, 2, 1, 3)

        x = x.reshape(
            batch_size * num_nodes,
            time_steps,
            hidden_dim,
        )

        # Temporal Transformer
        x = self.encoder(x)

        # Use the final temporal representation.
        x = x[:, -1, :]

        x = self.norm(x)

        # Restore:
        #
        # [B*N, H]
        #       ↓
        # [B, N, H]
        x = x.reshape(
            batch_size,
            num_nodes,
            hidden_dim,
        )

        return x